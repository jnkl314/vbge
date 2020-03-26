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

#include "DeepLabV3Plus_Inference_Settings.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

class VideoBackgroundEraser_Settings {
public:
    DeepLabV3Plus_Inference_Settings deeplabv3plus_inference;
};

} /* namespace VBGE */
#endif /* VIDEOBACKGROUNDERASER_SETTINGS_HPP_ */
