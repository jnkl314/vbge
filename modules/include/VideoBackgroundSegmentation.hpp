/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundSegmentation.hpp

 */
/*============================================================================*/

#ifndef VIDEOBACKGROUNDSEGMENTATION_HPP_
#define VIDEOBACKGROUNDSEGMENTATION_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "VideoBackgroundSegmentation_Settings.hpp"

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGS {


/*============================================================================*/
/* Forward Declaration                                                        */
/*============================================================================*/
class VideoBackgroundSegmentation_Algo;

/*============================================================================*/
/* Class Description                                                       */
/*============================================================================*/
/**
 * 	\brief       Class performing ...
 *
 */
/*============================================================================*/
class VideoBackgroundSegmentation {
private:
    std::unique_ptr<VideoBackgroundSegmentation_Algo> m_algo;

public:


    VideoBackgroundSegmentation(const VideoBackgroundSegmentation_Settings& i_settings);
    virtual ~VideoBackgroundSegmentation();

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
    int run(const cv::Mat& i_image, cv::Mat& o_backgroundMask);


};

} /* namespace VBGS */
#endif /* VIDEOBACKGROUNDSEGMENTATION_HPP_ */
