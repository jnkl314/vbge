/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepLabV3Plus_Inference_Settings.hpp

 */
/*============================================================================*/

#ifndef DEEPLABV3PLUS_INFERENCE_SETTINGS_HPP_
#define DEEPLABV3PLUS_INFERENCE_SETTINGS_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

class DeepLabV3Plus_Inference_Settings {
public:

    enum Strategy : int {Resize = 0, SlidingWindow = 1};

    std::string       model_path = "/some/path/data/best_deeplabv3plus_mobilenet_voc_os16.pt";
    int               background_classId = 0;
    cv::Size          inferenceSize = {513, 513};
    cv::Vec3f         model_mean = {0.485, 0.456, 0.406};
    cv::Vec3f         model_std = {0.229, 0.224, 0.225};
    Strategy          strategy = Resize;
    cv::Size          slidingWindow_overlap = {50, 50};
    torch::DeviceType inferenceDeviceType = torch::kCPU;
};

} /* namespace VBGE */
#endif /* DEEPLABV3PLUS_INFERENCE_SETTINGS_HPP_ */
