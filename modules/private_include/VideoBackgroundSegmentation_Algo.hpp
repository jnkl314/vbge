/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundSegmentation_Algo.hpp

 */
/*============================================================================*/

#ifndef VIDEOBACKGROUNDSEGMENTATION_ALGO_HPP_
#define VIDEOBACKGROUNDSEGMENTATION_ALGO_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "DeepLabV3Plus_Inference.hpp"
#include "VideoBackgroundSegmentation_Settings.hpp"

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGS {


class VideoBackgroundSegmentation_Algo {
public:

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run the text detection
     * @param[in] 		i_eastModelPath         : Path to a binary .pb containing the trained network
     * @param[in] 		i_confidenceThreshold	: Confidence threshold
     * @param[in] 		i_nmsThreshold          : Non-maximum suppression threshold
     *
     */
    /*============================================================================*/
    VideoBackgroundSegmentation_Algo(const VideoBackgroundSegmentation_Settings& i_settings);

    ~VideoBackgroundSegmentation_Algo();

    bool get_isInitialized();


    int run(const cv::Mat& i_image, cv::Mat& o_backgroundMask);

private:
    // Misc
    bool m_isInitialized = false;

    // Settings
    VideoBackgroundSegmentation_Settings m_settings;

    // Members
    DeepLabV3Plus_Inference m_deeplabv3plus_inference;


};

} /* namespace VBGS */
#endif /* VIDEOBACKGROUNDSEGMENTATION_ALGO_HPP_ */
