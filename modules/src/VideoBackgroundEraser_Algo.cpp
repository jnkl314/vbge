/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        VideoBackgroundEraser_Algo.cpp

 */
/*============================================================================*/

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <limits>
#include <iomanip>
#include <typeinfo>

#include "Utils_Logging.hpp"

#include "VideoBackgroundEraser_Algo.hpp"

/*============================================================================*/
/* Defines                                                                  */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {

VideoBackgroundEraser_Algo::VideoBackgroundEraser_Algo(const VideoBackgroundEraser_Settings &i_settings)
    : m_settings(i_settings),
      m_deeplabv3plus_inference(m_settings.deeplabv3plus_inference)
{

    if(false == m_deeplabv3plus_inference.get_isInitialized()) {
        logging_error("m_deeplabv3plus_inference was not correctly initialized.");
        return;
    }

    m_isInitialized = true;
}

VideoBackgroundEraser_Algo::~VideoBackgroundEraser_Algo()
{

}

bool VideoBackgroundEraser_Algo::get_isInitialized() {
    return m_isInitialized;
}


int VideoBackgroundEraser_Algo::run(const cv::Mat& i_image, cv::Mat& o_foregroundMask)
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

    // Run segmentation with DeepLabV3Plus to create a mask of the background
    cv::Mat backgroundMask;
    m_deeplabv3plus_inference.run(imageFloat, backgroundMask);

    cv::Mat image_uint8;
    imageFloat.convertTo(image_uint8, CV_8U, 255.);
    cv::cvtColor(image_uint8, image_uint8, cv::COLOR_BGR2GRAY);
//    cv::resize(image, image, cv::Size(), 0.5, 0.5);
//    cv::resize(backgroundMask, backgroundMask, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    cv::Mat foregroundDetection = 0 == backgroundMask;


    cv::erode(foregroundDetection, foregroundDetection, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)), cv::Point(-1, -1), 3);
    cv::dilate(foregroundDetection, foregroundDetection, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)), cv::Point(-1, -1), 3);

    static cv::Mat image_prev;
    static cv::Ptr<cv::DISOpticalFlow> optFLow = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
    constexpr int nbPQ = 1;
    constexpr int P[nbPQ] = {1};//, 1};
    constexpr int Q[nbPQ] = {3};//, 4};
    uint sumQ = 0;
    for(int i = 0 ; i < nbPQ ; ++i) {
        sumQ += Q[i];
    }
    static std::list<cv::Mat> detections_history;
    static cv::Mat statusMap;
    static cv::Mat flow;

    if(!image_prev.empty()) {

        // Compute optical flow betwen previous and current image
        optFLow->calc(image_uint8, image_prev, flow);

        // Convert flow to coordinates map
        cv::Mat mapXY;
        mapXY.create(flow.size(), flow.type());
        for(int y = 0 ; y < flow.rows ; ++y) {
            for(int x = 0 ; x < flow.cols ; ++x) {
                auto& f = flow.at<cv::Vec2f>(y, x);
                mapXY.at<cv::Vec2f>(y, x) = {f(0) + x, f(1) + y};
            }
        }

        // Remap past detections onto current referential
        for(auto& detections : detections_history) {
            cv::remap(detections, detections, mapXY, cv::noArray(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
        cv::remap(statusMap, statusMap, mapXY, cv::noArray(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Populate detections_history with new foregroundDetection
        {
            cv::Mat newDetection = cv::Mat::zeros(foregroundDetection.size(), CV_8U);
            newDetection.setTo(1, foregroundDetection);
            detections_history.push_front(newDetection);
            // Ensure we don't have too many detections
            while(detections_history.size() > sumQ) {
                detections_history.pop_back();
            }
        }

        // Sum detections, the value in sumDetections will be the number of times
        // this pixel was detected as foreground in the past images
        cv::Mat confirmedDetection;
        {
            cv::Mat sumDetections[2] = {cv::Mat::zeros(foregroundDetection.size(), CV_8U),
                                        cv::Mat::zeros(foregroundDetection.size(), CV_8U)};
            int idx = 0;
            int counter = 0;
            for(auto& detections : detections_history) {
                sumDetections[idx] = sumDetections[idx] + detections;
                if(++counter >= Q[idx]) {
                    ++idx;
                    counter = 0;
                }
            }

            // A detection is valid if it was seen more than P times in the last Q frames
            confirmedDetection = sumDetections[0] > P[0];
            for(int i = 1 ; i < nbPQ ; ++i) {
                confirmedDetection = confirmedDetection & (sumDetections[i] > P[i]);
            }
        }

        // Update status map
        statusMap.setTo(1, confirmedDetection);
        // Increase values of statusMap which were once confirmed but have not been seen recently
        cv::add(statusMap, 1, statusMap, (0 == confirmedDetection) & (0 != statusMap));
        // Remove the old confirmed values which have not been seen for too long
        statusMap.setTo(0, statusMap > 15);

        // Effective foreground is when statusMap is valid and we also add the current foreground detection
        o_foregroundMask = statusMap | foregroundDetection;

    } else {
        statusMap = cv::Mat::zeros(image_uint8.size(), CV_8U);

        o_foregroundMask = foregroundDetection;
    }

    image_uint8.copyTo(image_prev);


    return 0;
}

} /* namespace VBGE */
