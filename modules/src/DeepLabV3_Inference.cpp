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

    return 0;
}

} /* namespace VBGE */
