/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepLabV3_Inference.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "DeepLabV3_Inference.hpp"

/*============================================================================*/
/* Defines                                                                  */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

DeepLabV3_Inference::DeepLabV3_Inference(const DeepLabV3_Inference_Settings& i_settings)
    : m_settings(i_settings)
{
    m_model = torch::jit::load(m_settings.model_path, m_settings.inferenceDeviceType);

    // Check settings
    if(m_settings.FullSize != m_settings.strategy && m_settings.SlidingWindow != m_settings.strategy) {
        logging_error("m_settings.strategy has an unknown value (=" << m_settings.strategy << "). "
                      << "It should be 'FullSize'(=" << m_settings.FullSize << ").  or 'SlidingWindow'(=" << m_settings.SlidingWindow << "). ");
        return;
    }

    m_isInitialized = true;
}

DeepLabV3_Inference::~DeepLabV3_Inference()
{

}

bool DeepLabV3_Inference::get_isInitialized() {
    return m_isInitialized;
}


int DeepLabV3_Inference::run(const cv::Mat& i_image, cv::Mat& o_segmentation)
{
    if(false == get_isInitialized()) {
        logging_error("This instance was not correctly initialized.");
        return -1;
    }

    if(CV_32FC3 != i_image.type()) {
        logging_error("CV_32FC3 != i_image.type()");
        return -1;
    }

//    // Check on image size and change strategy if necessary
//    DeepLabV3_Inference_Settings::Strategy effectiveStrategy = m_settings.strategy;
//    if(DeepLabV3_Inference_Settings::SlidingWindow == effectiveStrategy &&
//            (i_image.rows < m_settings.slidingWindow_size.height || i_image.cols < m_settings.slidingWindow_size.width)) {
//        logging_warning("Image size (" << i_image.size() << ") is smaller in one or both dimension than the inference size (" << m_settings.inferenceSize << "). "
//                        << "Stragety forced to 'Resize'");
//        effectiveStrategy = DeepLabV3_Inference_Settings::Resize;
//    }

    switch(m_settings.strategy) {
    case DeepLabV3_Inference_Settings::FullSize: run_segmentation(i_image, o_segmentation); break;
    case DeepLabV3_Inference_Settings::SlidingWindow: run_window(i_image, o_segmentation); break;
    default:; // Can't get here because of the assert above
    }

    return 0;
}

void DeepLabV3_Inference::run_segmentation(const cv::Mat& i_image, cv::Mat& o_segmentation)
{
    // Normalize input image
    cv::Scalar meanValues(m_settings.model_mean);
    cv::Scalar stdValues(m_settings.model_std);
    cv::Mat imageNormalized = (i_image - meanValues)/stdValues;

    // We don't want to save the gradients during net.forward()
    torch::NoGradGuard no_grad_guard;

    // Prepare Input
    // Encapsulate i_src in a tensor (no deep copy) with OpenCV format NHWC
    std::vector<int64_t> srcSize = {1, i_image.rows, i_image.cols, i_image.channels()};
    std::vector<int64_t> srcStride = {1, static_cast<int64_t>(i_image.step1()), i_image.channels(), 1};
    torch::Tensor srcTensor_NHWC = torch::from_blob(i_image.data, srcSize, srcStride, torch::kCPU);
    // Permute format NHWC (OpenCV) to NCHW (PyTorch) (no deep copy)
    torch::Tensor inputTensor_NCHW = srcTensor_NHWC.permute({0, 3, 1, 2}).to(m_settings.inferenceDeviceType);

    // Inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor_NCHW);
    // /!\ Dynamic alloc
    torch::Tensor neuralNet_outputTensor_NCHW = m_model.forward(inputs).toTensor().squeeze(0); // Squeeze => remove first dimension, which was Batch size (=1)

    // Get the ID of the class with the max score
    torch::Tensor output_predictions = neuralNet_outputTensor_NCHW.argmax(0);
    // Convert from int64 to int32 to ease usage with OpenCV (there is no CV_64S)
    output_predictions = output_predictions.toType(torch::kInt32).to(torch::kCPU); // /!\ Dynamic alloc

    // Prepare output
    int height = output_predictions.sizes()[0];
    int width = output_predictions.sizes()[1];
    o_segmentation.create(height, width, CV_32S); // /!\ Dynamic alloc
    std::vector<int64_t> dstSize = {o_segmentation.rows, o_segmentation.cols};
    std::vector<int64_t> dstStride = {static_cast<int64_t>(o_segmentation.step1()), 1};
    torch::TensorOptions options;
    options = options.dtype(torch::kInt32);
    options = options.device(torch::kCPU);
    torch::Tensor segmentationTensor = torch::from_blob(o_segmentation.data, dstSize, dstStride, options);
    // Copy neuralNet_outputTensor_NHWC to outputTensor_NHWC
    segmentationTensor.copy_(output_predictions, true);
}

void DeepLabV3_Inference::create_windowList(const cv::Mat                &i_image,
                       const cv::Size                i_windowSize,
                       const cv::Size                i_overlapSize,
                             std::list<WindowImage> &o_windowList)
{
    CV_Assert(i_windowSize.width > i_overlapSize.width || i_windowSize.height > i_overlapSize.height);

    //** Compute ideal number of windows which minimize the overlap **//
    // Compute step
    int step_x = 0, N_x = 0;
    int step_y = 0, N_y = 0;
    // Along x, then y
    for(int i = 0 ; i < 2 ; ++i) {
        float w; // windowSize
        float L; // originSize
        float m; // minOverlap
        int *step, *N;
        if(0 == i) {
            L = i_image.cols;
            w = i_windowSize.width;
            m = i_overlapSize.width;
            step = &step_x;
            N = &N_x;
        } else {
            L = i_image.rows;
            w = i_windowSize.height;
            m = i_overlapSize.height;
            step = &step_y;
            N = &N_y;
        }

        if(L < w) {
            *N = 0;
        } else if(L == w) {
            *N = 1;
        } else {
            *N = std::ceil(2 + (L-2*(w-m))/(w-m*2));
            *step = std::floor((L-1-w)/(*N-1));
        }

    }

    // Create window list
    for(int ny = 0 ; ny < N_y ; ny++) {
        for(int nx = 0 ; nx < N_x ; nx++) {
            int x = nx*step_x;
            int y = ny*step_y;
            cv::Rect roi = cv::Rect(x, y, i_windowSize.width, i_windowSize.height);
            WindowImage wi;
            wi.im = i_image(roi).clone();
            wi.origin_in_source = cv::Point2i(x, y);
            o_windowList.push_back(wi);
        }
    }

}

void DeepLabV3_Inference::run_window(const cv::Mat& i_image, cv::Mat& o_segmentation)
{
    // Create list of windows
    std::list<WindowImage> windowList;
    create_windowList(i_image, m_settings.slidingWindow_size, m_settings.slidingWindow_overlap, windowList);

    // Create list of background masks and launch segmentation
    std::list<WindowImage> noBackgroundMask_windowList;
    for(auto& window : windowList) {
        cv::Mat noBackgroundMask_window;
        run_segmentation(window.im, noBackgroundMask_window);
        noBackgroundMask_windowList.push_back({noBackgroundMask_window, window.origin_in_source});
    }

    // Merge backgroundMask_windowList in o_backgroundMask
    o_segmentation.create(i_image.size(), CV_32S);
    o_segmentation.setTo(0);

    for(auto& backgroundMask_window : noBackgroundMask_windowList) {
        cv::Rect roi(backgroundMask_window.origin_in_source, backgroundMask_window.im.size());
        o_segmentation(roi) = cv::max(o_segmentation(roi), backgroundMask_window.im);
    }

    // Also run resize
    cv::Mat noBackgroundMask_resized;
    run_segmentation(i_image, noBackgroundMask_resized);
    o_segmentation = cv::max(o_segmentation, noBackgroundMask_resized);
}

} /* namespace VBGE */
