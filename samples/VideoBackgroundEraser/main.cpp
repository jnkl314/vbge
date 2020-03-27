/*============================================================================*/
/* File Description                                                           */
/*============================================================================*/
/**
 * @file        main.cpp

 */
/*============================================================================*/
#include <iostream>
#include <thread>
#include <cmath>
#include <chrono>
#include <ctime>

#include <tclap/CmdLine.h>

#include <Utils_Logging.hpp>
#include <VideoBackgroundEraser.hpp>

////// APPLICATION ARGUMENTS //////
struct {
    std::string inputPath;
    bool useCuda;
    bool hideDisplay;
    std::string outputPath;

    VBGE::VideoBackgroundEraser_Settings vbge_settings;

} typedef CmdArguments;

int initializeAndParseArguments(int argc, char **argv, CmdArguments& o_cmdArguments)
{
    ////*** Beginning of Arguments Handling ***////

    // Create and attach TCLAP arguments to cmd
    TCLAP::CmdLine cmd("TextDetection", ' ', "1.0");
    std::vector<std::shared_ptr<TCLAP::Arg> > tclap_args;
    // Add some custom parameter
    try {

//        cv::Vec3f         model_mean = {0.485, 0.456, 0.406};
//        cv::Vec3f         model_std = {0.229, 0.224, 0.225};
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("i", "inputPath",
                                                                                          "Path to video or a directory+pattern",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::SwitchArg             ("c", "useCuda",
                                                                                          "Use Cuda for inference",
                                                                                          cmd, false)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::SwitchArg             ("", "hideDisplay",
                                                                                          "Hide display of source image and result",
                                                                                          cmd, false)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("o", "outputPath",
                                                                                          "Path to a directory to save result",
                                                                                          false, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("m", "DeepLabV3PlusModelPath",
                                                                                          "Path to a PyTorch JIT binary .pb containing the trained model DeepLabV3Plus",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("n", "DeepImageMattingModelPath",
                                                                                          "Path to a PyTorch JIT binary .pb containing the trained model DeepImageMatting",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("b", "background_classId",
                                                                                         "ID of the background in the model",
                                                                                         false, 0, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("", "inferenceSize_width",
                                                                                         "Width of the inference for the given model",
                                                                                         false, 513, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("", "inferenceSize_height",
                                                                                         "Height of the inference for the given model",
                                                                                         false, 513, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::SwitchArg             ("s", "useSlidingWindow",
                                                                                          "Instead of resizing the input image to 513x513, use a sliding window and merge masks",
                                                                                          cmd, false)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("", "useSlidingWindow_overlapWidth",
                                                                                         "Height of the overlap between sliding windows",
                                                                                         false, 50, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("", "useSlidingWindow_overlapHeight",
                                                                                         "Width of the overlap between sliding windows",
                                                                                         false, 50, "int", cmd)));

    } catch(TCLAP::ArgException &e) {  // catch any exceptions
        logging_error("Failed to create TCLAP arguments" << std::endl <<
                      "TCLAP error: " << e.error() << " for arg " << e.argId());
        return -1;
    }

    // Parse all arguments
    try {
        cmd.setExceptionHandling(false);
        cmd.parse(argc, argv);
    } catch(TCLAP::ArgException &e) {
        logging_error("Failed to parse tclap arguments" << std::endl <<
                     "TCLAP error: " << e.error() << " for arg " << e.argId());
        return -1;
    } catch(TCLAP::ExitException &e) {
        exit(0);
    }

    ////*** End of Arguments Handling ***////

    // Dispatch arguments value in o_cmdArguments
    uint idx = 0;
    // Fetch custom argument value
    o_cmdArguments.inputPath   = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.useCuda     = dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue();
    o_cmdArguments.hideDisplay = dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue();
    o_cmdArguments.outputPath  = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();

    auto& deeplabv3plus = o_cmdArguments.vbge_settings.deeplabv3plus_inference;
    auto& deepimagematting = o_cmdArguments.vbge_settings.deepimagematting_inference;
    deeplabv3plus.model_path                    = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    deepimagematting.model_path                 = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    deeplabv3plus.background_classId            = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.inferenceSize.width           = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.inferenceSize.height          = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.strategy                      = true == dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue() ?
                VBGE::DeepLabV3Plus_Inference_Settings::SlidingWindow :
                VBGE::DeepLabV3Plus_Inference_Settings::Resize;
    deeplabv3plus.slidingWindow_overlap.width   = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.slidingWindow_overlap.height  = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.inferenceDeviceType           = o_cmdArguments.useCuda ? torch::kCUDA : torch::kCPU;
    deepimagematting.inferenceDeviceType        = deepimagematting.inferenceDeviceType;

    return 0;
}



////// MAIN //////
int main(int argc, char **argv)
{
    int res(0);

    CmdArguments cmdArguments;

    res = initializeAndParseArguments(argc, argv, cmdArguments);
    if(0 > res) {
        logging_error("initializeAndParseArguments() failed");
        return EXIT_FAILURE;
    }

    // Create and initialize VideoBackgroundEraser
    std::unique_ptr<VBGE::VideoBackgroundEraser> vbge(new VBGE::VideoBackgroundEraser(cmdArguments.vbge_settings));
    if(false == vbge->get_isInitialized()) {
        logging_error("VBGE::VideoBackgroundEraser was not correctly initialized");
        return EXIT_FAILURE;
    }

    // Open input
    logging_info("Open video/directory : " << cmdArguments.inputPath);
    cv::VideoCapture vc(cmdArguments.inputPath);

    if(!vc.isOpened()) {
        logging_error("Failed to open : " << cmdArguments.inputPath);
        return EXIT_FAILURE;
    }

    // Run Segmentation
    cv::Mat inputImage, outputImage;
    int cnt = 0;
    while(true)
    {
        // Load image
        logging_info("Grab next image, cnt = " << cnt);
        vc >> inputImage;
        if(inputImage.empty()) {
            logging_error("Failed to grab new image");
            break;
        } else {
//            cv::resize(inputImage, inputImage, cv::Size(), 0.2, 0.2);
            logging_info("Image of type " << cv::typeToString(inputImage.type()) << " and size " << inputImage.size());
        }

        // Process background segmentation and removal
        res = vbge->run(inputImage, outputImage);
        if(0 > res) {
            logging_error("VBGE::VideoBackgroundEraser::run() failed.");
            return EXIT_FAILURE;
        }

        // Save output
        if(!cmdArguments.outputPath.empty()) {
            logging_info("Saving results in : " << cmdArguments.outputPath);


            std::ostringstream oss;
            oss << cmdArguments.outputPath << "/" << std::setw(8) << std::setfill('0') << cnt++ << ".png";
            std::string path = oss.str();
            logging_info("Writing : " << path);
            cv::imwrite(path, outputImage);
        }

        // Display
        if(false == cmdArguments.hideDisplay) {
            cv::imshow("inputImage", inputImage);
            cv::imshow("outputImage", outputImage);
            int key = cv::waitKey(0) & 0xff;
            if(27 == key || 'q' == key) {
                break;
            }
        }
    }


    // Manually reset (and delete content of) pointer
    vbge.reset();

    return res;
}
