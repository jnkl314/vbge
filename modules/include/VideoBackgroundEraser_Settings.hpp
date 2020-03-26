/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundSegmentation_Settings.hpp

 */
/*============================================================================*/

#ifndef VIDEOBACKGROUNDSEGMENTATION_SETTINGS_HPP_
#define VIDEOBACKGROUNDSEGMENTATION_SETTINGS_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>

#include "DeepLabV3Plus_Inference_Settings.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGS {

class VideoBackgroundSegmentation_Settings {
public:
    DeepLabV3Plus_Inference_Settings deeplabv3plus_inference;
};

} /* namespace VBGS */
#endif /* VIDEOBACKGROUNDSEGMENTATION_SETTINGS_HPP_ */
