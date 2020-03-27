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
     * @brief         	Run the text detection
     * @param[in] 		i_eastModelPath         : Path to a binary .pb containing the trained network
     * @param[in] 		i_confidenceThreshold	: Confidence threshold
     * @param[in] 		i_nmsThreshold          : Non-maximum suppression threshold
     *
     */
    /*============================================================================*/
    DeepImageMatting_Inference(const DeepImageMatting_Inference_Settings& i_settings);

    ~DeepImageMatting_Inference();

    bool get_isInitialized();


    int run(const cv::Mat& i_image_rgba, cv::Mat& o_enhanced_image_rgba);

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
