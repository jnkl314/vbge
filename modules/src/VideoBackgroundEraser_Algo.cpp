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
      m_deeplabv3_inference(m_settings.deeplabv3_inference),
      m_deepimagematting_inference(m_settings.deepimagematting_inference)
{

    if(false == m_deeplabv3_inference.get_isInitialized()) {
        logging_error("m_deeplabv3_inference was not correctly initialized.");
        return;
    }

    m_optFLow = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);

    m_isInitialized = true;
}

VideoBackgroundEraser_Algo::~VideoBackgroundEraser_Algo()
{

}

bool VideoBackgroundEraser_Algo::get_isInitialized() {
    return m_isInitialized;
}

int VideoBackgroundEraser_Algo::run(const cv::Mat& i_image, cv::Mat& o_image_withoutBackground)
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

    /*
    // Apply local contrast enhancement
    cv::Mat src;
    imageFloat.convertTo(src, CV_8U, 255.);
    cv::Mat image_lce; // lce for local contrast enhancement
    {
        // Switch to HSV
        cv::Mat hsv;
        cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

        // Split channels
        std::vector<cv::Mat> hsv_split;
        cv::split(hsv, hsv_split);

        // Work on V
        cv::Mat lum = hsv_split[2];
        {
            // Apply local histogram equalization
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(40, cv::Size(16, 16));
            cv::Mat enhancedLum;
            clahe->apply(lum, enhancedLum);

            hsv_split[2] = enhancedLum;
        }

        // Merge back channels
        cv::merge(hsv_split, hsv);

        // Switch back to BGR
        cv::cvtColor(hsv, image_lce, cv::COLOR_HSV2BGR);
    }
    cv::Mat imageFloat_lce;
    image_lce.convertTo(imageFloat_lce, CV_32F, 1./255.);
    //*/

    // Run segmentation with DeepLabV3 to create a mask of the background
    cv::Mat segmentation;
    m_deeplabv3_inference.run(imageFloat, segmentation);

    // Debug display
//    {
//        cv::Mat segmentation_uint8;
//        segmentation.convertTo(segmentation_uint8, CV_8U, 50);
//        cv::Mat segmentationColor;
//        cv::applyColorMap(segmentation_uint8, segmentationColor, cv::COLORMAP_HSV);
//        cv::imshow("segmentationColor", segmentationColor);
//    }
//    {
//        cv::Mat segmentation_uint8;
//        segmentation.convertTo(segmentation_uint8, CV_8U);
//        cv::cvtColor(segmentation_uint8, segmentation_uint8, cv::COLOR_GRAY2BGR);
//        cv::imshow("segmentation_uint8", segmentation_uint8);
//    }

    cv::Mat backgroundMask;
    if(m_settings.deeplabv3_inference.background_classId_vector.empty()) {
        logging_error("m_settings.deeplabv3_inference.background_classId_vector is empty.");
        return -1;
    }
    for(auto& background_id : m_settings.deeplabv3_inference.background_classId_vector) {
        if(backgroundMask.empty()) {
            backgroundMask = (segmentation == background_id);
        } else {
            backgroundMask = backgroundMask | (segmentation == background_id);
        }
    }

//    imageFloat.copyTo(o_image_withoutBackground);
//    o_image_withoutBackground.setTo(0, backgroundMask);
//    return 0;

    // Prepare intermediary data
    cv::Mat foregroundMask;
    cv::Mat image_rgb_uint8;
    imageFloat.convertTo(image_rgb_uint8, CV_8U, 255.);

    // Run temporal processing to try and keep consistency between successive frames
    if(m_settings.enable_temporalManagement) {
        if(0 > temporalManagement(image_rgb_uint8, backgroundMask, foregroundMask)) {
            logging_error("temporalManagement() failed.");
            return -1;
        }
    } else {
        foregroundMask = 0 == backgroundMask;
    }

    // Generate trimap
    cv::Mat trimap;
    compute_trimap(image_rgb_uint8, foregroundMask, trimap);
    // Convert image rgb with trimap to make a rgba image
    cv::Mat imageFloat_rgba;
    std::vector<cv::Mat> image_rgba_planar;
    cv::split(imageFloat, image_rgba_planar);
    image_rgba_planar.push_back(cv::Mat());
    trimap.convertTo(image_rgba_planar.back(), CV_32F, 1./255.);
    cv::merge(image_rgba_planar, imageFloat_rgba);


    // Run Deep Image Matting
    cv::Mat alpha_prediction;
    m_deepimagematting_inference.run(imageFloat_rgba, alpha_prediction);

    // Post process alpha_prediction
    alpha_prediction.setTo(0, 0 == trimap);
    alpha_prediction.setTo(255, 255 == trimap);

    // Replace alpha in image_rgba_planar with alpha_prediction
//    alpha_prediction.convertTo(image_rgba_planar[3], CV_8U, 255.);
    image_rgba_planar[3] = alpha_prediction;
    cv::Mat image_withoutBackground;
    cv::merge(image_rgba_planar, image_withoutBackground);

    // Facultative conversion to the same depth as input
    if(CV_32F != i_image.depth()) {
        switch(i_image.depth()) {
        case CV_8U: image_withoutBackground.convertTo(o_image_withoutBackground, CV_8U, 255.); break;
        case CV_16U: image_withoutBackground.convertTo(o_image_withoutBackground, CV_16U, 65535.); break;
        default:
            logging_error("Unsuported input image depth (" << cv::typeToString(i_image.depth()) << "). Supported depths are CV_32F, CV_16U and CV_8U");
            return -1;
        }
    } else {
        o_image_withoutBackground = image_withoutBackground;
    }

    return 0;
}


int VideoBackgroundEraser_Algo::temporalManagement(const cv::Mat& i_image_rgb_uint8, const cv::Mat& i_backgroundMask, cv::Mat& o_foregroundMask)
{
    cv::Mat image_uint8;
    cv::cvtColor(i_image_rgb_uint8, image_uint8, cv::COLOR_BGR2GRAY);
    cv::Mat foregroundDetection = 0 == i_backgroundMask;

    constexpr int nbPQ = 2;
    constexpr int P[nbPQ] = {1, 2};
    constexpr int Q[nbPQ] = {2, 8};
    uint sumQ = 0;
    for(int i = 0 ; i < nbPQ ; ++i) {
        sumQ += Q[i];
    }

    if(!m_image_prev.empty()) {

        // Compute optical flow betwen previous and current image
        m_optFLow->calc(image_uint8, m_image_prev, m_flow);

        // Convert flow to coordinates map
        m_mapXY.create(m_flow.size(), m_flow.type());
        for(int y = 0 ; y < m_flow.rows ; ++y) {
            for(int x = 0 ; x < m_flow.cols ; ++x) {
                auto& f = m_flow.at<cv::Vec2f>(y, x);
                m_mapXY.at<cv::Vec2f>(y, x) = {f(0) + x, f(1) + y};
            }
        }

        // Remap past detections onto current referential
        for(auto& detections : m_detections_history) {
            cv::remap(detections, detections, m_mapXY, cv::noArray(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
        cv::remap(m_statusMap, m_statusMap, m_mapXY, cv::noArray(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Populate detections_history with new foregroundDetection
        {
            cv::Mat newDetection = cv::Mat::zeros(foregroundDetection.size(), CV_8U);
            newDetection.setTo(1, foregroundDetection);
            m_detections_history.push_front(newDetection);
            // Ensure we don't have too many detections
            while(m_detections_history.size() > sumQ) {
                m_detections_history.pop_back();
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
            for(auto& detections : m_detections_history) {
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
        m_statusMap.setTo(1, confirmedDetection);
        // Increase values of statusMap which were once confirmed but have not been seen recently
        cv::add(m_statusMap, 1, m_statusMap, (0 == confirmedDetection) & (0 != m_statusMap));
        // Remove the old confirmed values which have not been seen for too long
        m_statusMap.setTo(0, m_statusMap > 20);

        // Effective foreground is when statusMap is valid and we also add the current foreground detection
        o_foregroundMask = (0 != m_statusMap);

    } else {
        m_statusMap = cv::Mat::zeros(image_uint8.size(), CV_8U);

        o_foregroundMask.create(foregroundDetection.size(), CV_8U);
        o_foregroundMask.setTo(0);
    }

    image_uint8.copyTo(m_image_prev);

    return 0;
}

void VideoBackgroundEraser_Algo::compute_trimap(const cv::Mat& i_image, const cv::Mat& i_foreground, cv::Mat &o_trimap)
{
    // Generate standard trimap with morpho maths
    constexpr int kSize = 3;
    constexpr int dilate_iterations = 2;
    constexpr int erode_iterations = 7;
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kSize, kSize));
    cv::Mat dilated;
    cv::dilate(i_foreground, dilated, kernel, cv::Point(-1, -1), dilate_iterations);
    cv::Mat eroded;
    cv::erode(i_foreground, eroded, kernel, cv::Point(-1, -1), erode_iterations);

    o_trimap.create(i_foreground.size(), CV_8U);
    o_trimap.setTo(128);
    o_trimap.setTo(255, eroded >= 255);
    o_trimap.setTo(0, dilated <= 0);

    return ;
    // Only works for homogeneous background, like sky etc.
    // Improve trimap with image content
    // Compute mask for an area around the foreground
    cv::Mat foregroundContour;
    cv::dilate(i_foreground, foregroundContour, kernel, cv::Point(-1, -1), 10*dilate_iterations);
    foregroundContour = foregroundContour & (0 == i_foreground);

    // Compute background mean and standard deviation in a large area around the current foreground
    cv::Vec3d backgroundMean, backgroundStD;
    cv::meanStdDev(i_image, backgroundMean, backgroundStD, foregroundContour);

    // Also compute local mean and StD over a small area
    cv::Mat localMean, localStD;
    {
        const cv::Size localAreaSize = {11, 11};

        cv::Mat image32f;
        i_image.convertTo(image32f, CV_32F);

        cv::blur(image32f, localMean, localAreaSize);
        cv::Mat squareLocalMean;
        cv::blur(image32f.mul(image32f), squareLocalMean, localAreaSize);
        cv::sqrt(squareLocalMean - localMean.mul(localMean), localStD);
    }

    // Inefficient double loop, but that will do the job for now
    // Compare local and global mean/std in "confirmed" area of the trimap
    for(int y = 0 ; y < i_image.rows ; ++y) {
        for(int x = 0 ; x < i_image.cols ; ++x) {
            if(255 == o_trimap.at<uint8_t>(y, x)) {
                // Check if the local mean is enclosed in the the global mean +/- StD
                // Also check that the local StD is fairly low
                cv::Vec3f& lmu = localMean.at<cv::Vec3f>(y, x);
                cv::Vec3f& lstd = localStD.at<cv::Vec3f>(y, x);
                bool isUncertain = true;
                for(int c = 0 ; c < 3 ; ++c) {
                    if(5 < lstd(c) || std::abs(lmu(c) - backgroundMean(c)) > 0.5*backgroundStD(c)) {
                        isUncertain = false;
                    }
                }
                if(isUncertain) {
                    o_trimap.at<uint8_t>(y, x) = 128;
                }

            }
        }
    }
}

} /* namespace VBGE */
