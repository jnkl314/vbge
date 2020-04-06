/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepLabV3_Inference.hpp

 */
/*============================================================================*/

#ifndef DEEPLABV3_INFERENCE_HPP_
#define DEEPLABV3_INFERENCE_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "DeepLabV3_Inference_Settings.hpp"

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {


class DeepLabV3_Inference {
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
    DeepLabV3_Inference(const DeepLabV3_Inference_Settings& i_settings);

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Destructor
     *
     */
    /*============================================================================*/
    ~DeepLabV3_Inference();

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
     * @param[in] 		i_image        : Input image, RGB packed, 3-float32 (CV_32FC3)
     * @param[out]		o_segmentation : Output image, classes id in int32, same size as i_image
     *
     */
    /*============================================================================*/
    int run(const cv::Mat& i_image, cv::Mat& o_segmentation);

private:
    // Misc
    bool m_isInitialized = false;

    // Members
    torch::jit::script::Module m_model;

    // Settings
    const DeepLabV3_Inference_Settings m_settings;
};

} /* namespace VBGE */
#endif /* DEEPLABV3_INFERENCE_HPP_ */
