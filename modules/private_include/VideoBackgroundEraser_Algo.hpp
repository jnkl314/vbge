/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundEraser_Algo.hpp

 */
/*============================================================================*/

#ifndef VIDEOBACKGROUNDERASER_ALGO_HPP_
#define VIDEOBACKGROUNDERASER_ALGO_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "DeepLabV3Plus_Inference.hpp"
#include "VideoBackgroundEraser_Settings.hpp"

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {


class VideoBackgroundEraser_Algo {
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
    VideoBackgroundEraser_Algo(const VideoBackgroundEraser_Settings& i_settings);

    ~VideoBackgroundEraser_Algo();

    bool get_isInitialized();


    int run(const cv::Mat& i_image, cv::Mat& o_foregroundMask);

private:
    // Misc
    bool m_isInitialized = false;

    // Settings
    VideoBackgroundEraser_Settings m_settings;

    // Members
    DeepLabV3Plus_Inference m_deeplabv3plus_inference;

    cv::Mat m_image_prev;
    cv::Ptr<cv::DISOpticalFlow> m_optFLow;
    std::list<cv::Mat> m_detections_history;
    cv::Mat m_statusMap;
    cv::Mat m_flow;
    cv::Mat m_mapXY;


};

} /* namespace VBGE */
#endif /* VIDEOBACKGROUNDERASER_ALGO_HPP_ */
