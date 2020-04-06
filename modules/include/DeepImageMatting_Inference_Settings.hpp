/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepImageMatting_Inference_Settings.hpp

 */
/*============================================================================*/

#ifndef DEEPIMAGEMATTING_INFERENCE_SETTINGS_HPP_
#define DEEPIMAGEMATTING_INFERENCE_SETTINGS_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

class DeepImageMatting_Inference_Settings {
public:

    //! @brief Path to a PyTorch JIT binary .pb containing the trained model DeepImageMatting
    std::string       model_path = "/some/path/data/best_DeepImageMatting.pt";

    //! @brief Mean value of the dataset on which DeepImageMatting feature extractor (resnet101) was trained
    cv::Vec3f         model_mean = {0.485, 0.456, 0.406};

    //! @brief Standard Deviation value of the dataset on which DeepImageMatting feature extractor (resnet101) was trained
    cv::Vec3f         model_std = {0.229, 0.224, 0.225};

    //! @brief Device to use for inference : torch::kCPU or torch::kCUDA (multi GPU is not handled)
    torch::DeviceType inferenceDeviceType = torch::kCPU;
};

} /* namespace VBGE */
#endif /* DEEPIMAGEMATTING_INFERENCE_SETTINGS_HPP_ */
