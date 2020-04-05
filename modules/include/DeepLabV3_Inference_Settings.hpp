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

    enum Strategy : int {FullSize = 0, SlidingWindow = 1};

    std::string          model_path = "/some/path/data/best_deeplabv3_skydiver.pt";
    std::vector<int32_t> background_classId_vector = {0, 3, 4};
    cv::Vec3f            model_mean = {0.485, 0.456, 0.406};
    cv::Vec3f            model_std = {0.229, 0.224, 0.225};
    Strategy             strategy = FullSize;
    cv::Size             slidingWindow_size = {224, 224};
    cv::Size             slidingWindow_overlap = {50, 50};
    torch::DeviceType    inferenceDeviceType = torch::kCPU;
};

} /* namespace VBGE */
#endif /* DEEPLABV3_INFERENCE_SETTINGS_HPP_ */
