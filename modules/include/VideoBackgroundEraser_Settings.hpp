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
    //! @brief Settings class for the inference encapsulation of DeepLabV3
    DeepLabV3_Inference_Settings deeplabv3_inference;

    //! @brief Settings class for the inference encapsulation of DeepImageMatting
    DeepImageMatting_Inference_Settings deepimagematting_inference;

    //! @brief "Enable temporal management of scene to improve accuracy between frames. Might not work well for video where the background is moving
    bool enable_temporalManagement = false;

    //! @brief Rescale factor for Deep Image Matting
    float imageMatting_scale = 1.f;
};

} /* namespace VBGE */
#endif /* VIDEOBACKGROUNDERASER_SETTINGS_HPP_ */
