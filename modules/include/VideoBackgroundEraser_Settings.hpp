/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundEraser_Settings.hpp

 */
/*============================================================================*/

#ifndef VIDEOBACKGROUNDERASER_SETTINGS_HPP_
#define VIDEOBACKGROUNDERASER_SETTINGS_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>

#include "DeepLabV3_Inference_Settings.hpp"
#include "DeepImageMatting_Inference_Settings.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

class VideoBackgroundEraser_Settings {
public:
    DeepLabV3_Inference_Settings deeplabv3_inference;
    DeepImageMatting_Inference_Settings deepimagematting_inference;
    bool enable_temporalManagement = false;
    float imageMatting_scale = 1.f;
};

} /* namespace VBGE */
#endif /* VIDEOBACKGROUNDERASER_SETTINGS_HPP_ */
