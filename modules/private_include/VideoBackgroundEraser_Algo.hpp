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

#include "DeepLabV3_Inference.hpp"
#include "DeepImageMatting_Inference.hpp"
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
     * @brief         	Constructor
     * @param[in] 		i_settings         : user settings
     *
     */
    /*============================================================================*/
    VideoBackgroundEraser_Algo(const VideoBackgroundEraser_Settings& i_settings);

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Destructor
     *
     */
    /*============================================================================*/
    ~VideoBackgroundEraser_Algo();

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Class instance status
     * @return 		(bool)         : True if the class instance was correctly initialized
     *
     */
    /*============================================================================*/
    bool get_isInitialized();

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Perform inference of DeepLabV3
     * @param[in] 		i_image                   : Input image, RGB packed, CV_8UC3, CV_16UC3 or CV_32FC3
     * @param[out]		o_image_withoutBackground : Output image, RGBA packed, same size as i_image but with 4 channels
     *
     */
    /*============================================================================*/
    int run(const cv::Mat& i_image, cv::Mat& o_image_withoutBackground);

private:
    // Misc
    bool m_isInitialized = false;

    // Settings
    VideoBackgroundEraser_Settings m_settings;

    // Members
    DeepLabV3_Inference m_deeplabv3_inference;
    cv::Mat m_image_prev;
    cv::Ptr<cv::DISOpticalFlow> m_optFLow;
    std::list<cv::Mat> m_detections_history;
    cv::Mat m_statusMap;
    cv::Mat m_flow;
    cv::Mat m_mapXY;
    DeepImageMatting_Inference m_deepimagematting_inference;

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Improve accuracy of foreground objects over time
     * @param[in] 		i_image_rgb_uint8 : Input image, RGB packed, CV_8UC3
     * @param[in] 		i_backgroundMask  : Input mask, CV_8UC1. 255 for background pixels, 0 for the foreground
     * @param[out]		o_foregroundMask  : Output mask, CV_8UC1, 255 for foreground pixels, 0 for the background
     *
     */
    /*============================================================================*/
    int temporalManagement(const cv::Mat& i_image_rgb_uint8, const cv::Mat& i_backgroundMask, cv::Mat& o_foregroundMask);

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Compute trimap based on the foreground mask
     * @param[in] 		i_foreground : Input mask, CV_8UC1. 255 for foreground pixels, 0 for the background
     * @param[out]		o_trimap     : Output mask, CV_8UC1, 255 for foreground pixels, 128 for uncertain areas, 0 for the background
     *
     */
    /*============================================================================*/
    void compute_trimap(const cv::Mat& i_foreground, cv::Mat &o_trimap);

};

} /* namespace VBGE */
#endif /* VIDEOBACKGROUNDERASER_ALGO_HPP_ */
