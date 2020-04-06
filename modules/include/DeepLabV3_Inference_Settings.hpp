/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepLabV3_Inference_Settings.hpp

 */
/*============================================================================*/

#ifndef DEEPLABV3_INFERENCE_SETTINGS_HPP_
#define DEEPLABV3_INFERENCE_SETTINGS_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

class DeepLabV3_Inference_Settings {
public:

    //! @brief Path to a PyTorch JIT binary .pb containing the trained model DeepLabV3
    std::string          model_path = "/some/path/data/best_deeplabv3_skydiver.pt";

    //! @brief IDs of the background in the model. For example 0 for no-label, 3 for ground and 4 for sky
    std::vector<int32_t> background_classId_vector = {0, 3, 4};

    //! @brief Mean value of the dataset on which DeepLabV3 feature extractor (resnet101) was trained
    cv::Vec3f            model_mean = {0.485, 0.456, 0.406};

    //! @brief Standard Deviation value of the dataset on which DeepLabV3 feature extractor (resnet101) was trained
    cv::Vec3f            model_std = {0.229, 0.224, 0.225};

    //! @brief Device to use for inference : torch::kCPU or torch::kCUDA (multi GPU is not handled)
    torch::DeviceType    inferenceDeviceType = torch::kCPU;
};

} /* namespace VBGE */
#endif /* DEEPLABV3_INFERENCE_SETTINGS_HPP_ */
