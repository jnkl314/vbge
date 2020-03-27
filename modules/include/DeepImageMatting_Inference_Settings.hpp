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

    std::string       model_path = "/some/path/data/best_DeepImageMatting.pt";
    cv::Vec3f         model_mean = {0.485, 0.456, 0.406};
    cv::Vec3f         model_std = {0.229, 0.224, 0.225};
    torch::DeviceType inferenceDeviceType = torch::kCPU;
};

} /* namespace VBGE */
#endif /* DEEPIMAGEMATTING_INFERENCE_SETTINGS_HPP_ */
