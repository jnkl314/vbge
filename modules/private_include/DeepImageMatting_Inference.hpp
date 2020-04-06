/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        DeepImageMatting_Inference.hpp

 */
/*============================================================================*/

#ifndef DEEPIMAGEMATTING_INFERENCE_HPP_
#define DEEPIMAGEMATTING_INFERENCE_HPP_

/*============================================================================*/
/* Includes                                                                   */
/*============================================================================*/
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "DeepImageMatting_Inference_Settings.hpp"

/*============================================================================*/
/* define                                                                     */
/*============================================================================*/

/*============================================================================*/
/* namespace                                                                  */
/*============================================================================*/
namespace VBGE {


class DeepImageMatting_Inference {
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
    DeepImageMatting_Inference(const DeepImageMatting_Inference_Settings& i_settings);

    /*============================================================================*/
    /* Function Description                                                       */
    /*============================================================================*/
    /**
     * @brief         	Destructor
     *
     */
    /*============================================================================*/
    ~DeepImageMatting_Inference();

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
     * @brief         	Perform inference of DeepImageMatting
     * @param[in] 		i_image            : Input image, RGBA packed, 4-float32 (CV_32FC4)
     * @param[out]		o_alpha_prediction : Output image, alpha component (float32 -> CV_32F), same size as i_image
     *
     */
    /*============================================================================*/
    int run(const cv::Mat& i_image_rgba, cv::Mat& o_alpha_prediction);

private:
    // Misc
    bool m_isInitialized = false;

    // Members
    torch::jit::script::Module m_model;

    // Settings
    const DeepImageMatting_Inference_Settings m_settings;
};

} /* namespace VBGE */
#endif /* DEEPIMAGEMATTING_INFERENCE_HPP_ */
