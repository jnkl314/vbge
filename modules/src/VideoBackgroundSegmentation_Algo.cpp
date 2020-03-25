/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundSegmentation_Algo.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "VideoBackgroundSegmentation_Algo.hpp"

/*============================================================================*/
/* Defines                                                                  */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGS {

VideoBackgroundSegmentation_Algo::VideoBackgroundSegmentation_Algo(const VideoBackgroundSegmentation_Settings &i_settings)
    : m_settings(i_settings),
      m_deeplabv3plus_inference(m_settings.deeplabv3plus_inference)
{

    if(false == m_deeplabv3plus_inference.get_isInitialized()) {
        logging_error("m_deeplabv3plus_inference was not correctly initialized.");
        return;
    }

    m_isInitialized = true;
}

VideoBackgroundSegmentation_Algo::~VideoBackgroundSegmentation_Algo()
{

}

bool VideoBackgroundSegmentation_Algo::get_isInitialized() {
    return m_isInitialized;
}


int VideoBackgroundSegmentation_Algo::run(const cv::Mat& i_image, cv::Mat& o_backgroundMask)
{
    if(false == get_isInitialized()) {
        logging_error("This instance was not correctly initialized.");
        return -1;
    }

    cv::Mat imageFloat;
    if(CV_32F != i_image.depth()) {
        switch(i_image.depth()) {
        case CV_8U: i_image.convertTo(imageFloat, CV_32F, 1./255.); break;
        case CV_16U: i_image.convertTo(imageFloat, CV_32F, 1./65535.); break;
        default:
            logging_error("Unsuported input image depth (" << cv::typeToString(i_image.depth()) << "). Supported depths are CV_32F, CV_16U and CV_8U");
            return -1;
        }
    } else {
        imageFloat = i_image;
    }

    CV_Assert(CV_32FC3 == imageFloat.type());

    m_deeplabv3plus_inference.run(imageFloat, o_backgroundMask);

    return 0;
}

} /* namespace VBGS */
