/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundEraser.hpp

 */
/*============================================================================*/

#ifndef VIDEOBACKGROUNDERASER_HPP_
#define VIDEOBACKGROUNDERASER_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "VideoBackgroundEraser_Settings.hpp"

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {


/*============================================================================*/
/* Forward Declaration                                                        */
/*============================================================================*/
class VideoBackgroundEraser_Algo;

/*============================================================================*/
/* Class Description                                                       */
/*============================================================================*/
/**
 * 	\brief       Class performing ...
 *
 */
/*============================================================================*/
class VideoBackgroundEraser {
private:
    std::unique_ptr<VideoBackgroundEraser_Algo> m_algo;

public:


    VideoBackgroundEraser(const VideoBackgroundEraser_Settings& i_settings);
    virtual ~VideoBackgroundEraser();

    bool get_isInitialized();

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Run the text detection
     * @param[in] 		i_src                       : Input image to process
     * @param[out] 		o_detectedText_boundingBox	: Vector of bounding box of the detected text area
     *
     */
    /*============================================================================*/
    int run(const cv::Mat& i_image, cv::Mat& o_image_withoutBackground);


};

} /* namespace VBGE */
#endif /* VIDEOBACKGROUNDERASER_HPP_ */
