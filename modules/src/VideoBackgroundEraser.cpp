/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundEraser.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "VideoBackgroundEraser.hpp"
#include "VideoBackgroundEraser_Algo.hpp"

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

VideoBackgroundEraser::VideoBackgroundEraser(const VideoBackgroundEraser_Settings& i_settings)
{
    m_algo.reset(new VideoBackgroundEraser_Algo(i_settings));
}

VideoBackgroundEraser::~VideoBackgroundEraser()
{
    m_algo.reset();
}

bool VideoBackgroundEraser::get_isInitialized()
{
    return m_algo->get_isInitialized();
}

int VideoBackgroundEraser::run(const cv::Mat& i_image, cv::Mat& o_foregroundMask)
{
    if(false == m_algo->get_isInitialized()) {
        logging_error("This instance was not correctly initialized.");
        return -1;
    }

    // Tests on i_src
    if(i_image.empty()) {
        logging_error("i_image is empty.");
        return -1;
    }
    if(3 != i_image.channels()) {
        logging_error("i_image does not have 3 channels.");
        return -1;
    }

    // Actual call to algorithm
    if(0 > m_algo->run(i_image, o_foregroundMask)) {
        logging_error("m_algo->run() failed.");
        return -1;
    }

    return 0;
}

} /* namespace VBGE */
