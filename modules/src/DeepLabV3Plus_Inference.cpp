/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepLabV3Plus_Inference.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "DeepLabV3Plus_Inference.hpp"

/*============================================================================*/
/* Defines                                                                  */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGS {

DeepLabV3Plus_Inference::DeepLabV3Plus_Inference(const DeepLabV3Plus_Inference_Settings& i_settings)
    : m_settings(i_settings)
{
    m_model = torch::jit::load(m_settings.model_path, m_settings.inferenceDeviceType);

    // Check settings
    if(m_settings.Resize != m_settings.strategy && m_settings.SlidingWindow != m_settings.strategy) {
        logging_error("m_settings.strategy has an unknown value (=" << m_settings.strategy << "). "
                      << "It should be 'Resize'(=" << m_settings.Resize << ").  or 'SlidingWindow'(=" << m_settings.SlidingWindow << "). ");
        return;
    }

    m_isInitialized = true;
}

DeepLabV3Plus_Inference::~DeepLabV3Plus_Inference()
{

}

bool DeepLabV3Plus_Inference::get_isInitialized() {
    return m_isInitialized;
}


int DeepLabV3Plus_Inference::run(const cv::Mat& i_image, cv::Mat& o_backgroundMask)
{
    if(false == get_isInitialized()) {
        logging_error("This instance was not correctly initialized.");
        return -1;
    }

    if(CV_32FC3 != i_image.type()) {
        logging_error("CV_32FC3 != i_image.type()");
        return -1;
    }

    cv::Mat noBackground;
    switch(m_settings.strategy) {
    case DeepLabV3Plus_Inference_Settings::Resize: run_resize(i_image, noBackground); break;
    case DeepLabV3Plus_Inference_Settings::SlidingWindow: run_window(i_image, noBackground); break;
    default:; // Can't get here because of the assert above
    }

    o_backgroundMask = 0 == noBackground;

    return 0;
}

void DeepLabV3Plus_Inference::segmentBackground(const cv::Mat& i_image, cv::Mat& o_noBackgroundMask)
{
    // Checks
    CV_Assert(m_settings.inferenceSize == i_image.size());

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
    cv::Mat segmentation(height, width, CV_32S); // /!\ Dynamic alloc
    std::vector<int64_t> dstSize = {segmentation.rows, segmentation.cols};
    std::vector<int64_t> dstStride = {static_cast<int64_t>(segmentation.step1()), 1};
    torch::Tensor segmentationTensor = torch::from_blob(segmentation.data, dstSize, dstStride, torch::kCPU);
    // Copy neuralNet_outputTensor_NHWC to outputTensor_NHWC
    segmentationTensor.copy_(output_predictions, true);

    // Build mask, valid for every pixel segmented as background
    o_noBackgroundMask = m_settings.background_classId != segmentation;
}

void DeepLabV3Plus_Inference::run_resize(const cv::Mat& i_image, cv::Mat& o_noBackgroundMask)
{
    if(i_image.size() != m_settings.inferenceSize) {
        cv::Mat image_resized, noBackgroundMask_resized;;
        cv::resize(i_image, image_resized, m_settings.inferenceSize, 0, 0, cv::INTER_CUBIC);
        segmentBackground(image_resized, noBackgroundMask_resized);
        cv::resize(noBackgroundMask_resized, o_noBackgroundMask, i_image.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        segmentBackground(i_image, o_noBackgroundMask);
    }
}

void DeepLabV3Plus_Inference::create_windowList(const cv::Mat                &i_image,
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

void DeepLabV3Plus_Inference::run_window(const cv::Mat& i_image, cv::Mat& o_noBackgroundMask)
{
    // Create list of windows
    std::list<WindowImage> windowList;
    create_windowList(i_image, m_settings.inferenceSize, m_settings.slidingWindow_overlap, windowList);

    // Create list of background masks and launch segmentation
    std::list<WindowImage> noBackgroundMask_windowList;
    for(auto& window : windowList) {
        cv::Mat noBackgroundMask_window;
        segmentBackground(window.im, noBackgroundMask_window);
        noBackgroundMask_windowList.push_back({noBackgroundMask_window, window.origin_in_source});
    }

    // Merge backgroundMask_windowList in o_backgroundMask
    o_noBackgroundMask.create(i_image.size(), CV_8U);
    o_noBackgroundMask.setTo(0);

    for(auto& backgroundMask_window : noBackgroundMask_windowList) {
        cv::Rect roi(backgroundMask_window.origin_in_source, backgroundMask_window.im.size());
        o_noBackgroundMask(roi) = cv::max(o_noBackgroundMask(roi), backgroundMask_window.im);
    }

    // Also run resize
    cv::Mat noBackgroundMask_resized;
    run_resize(i_image, noBackgroundMask_resized);
    o_noBackgroundMask = cv::max(o_noBackgroundMask, noBackgroundMask_resized);
}

} /* namespace VBGS */
