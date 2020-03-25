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
#include <VideoBackgroundSegmentation.hpp>

////// APPLICATION ARGUMENTS //////
struct {
    std::string inputPath;
    std::string outputPath;
    bool useCuda;

    VBGS::VideoBackgroundSegmentation_Settings vbgs_settings;

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
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("o", "outputPath",
                                                                                          "Path to a directory to save result",
                                                                                          false, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("m", "DeepLabV3PlusModelPath",
                                                                                          "Path to a PyTorch JIT binary .pb containing the trained model DeepLabV3Plus",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("b", "background_classId",
                                                                                         "ID of the background in the model",
                                                                                         false, 0, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("w", "inferenceSize_width",
                                                                                         "Width of the inference for the given model",
                                                                                         false, 513, "int", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<int>       ("h", "inferenceSize_height",
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
    o_cmdArguments.inputPath  = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.outputPath = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.useCuda    = dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue();

    auto& deeplabv3plus = o_cmdArguments.vbgs_settings.deeplabv3plus_inference;
    deeplabv3plus.model_path                    = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    deeplabv3plus.background_classId            = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.inferenceSize.width           = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.inferenceSize.height          = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.strategy                      = true == dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue() ?
                VBGS::DeepLabV3Plus_Inference_Settings::SlidingWindow :
                VBGS::DeepLabV3Plus_Inference_Settings::Resize;
    deeplabv3plus.slidingWindow_overlap.width   = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.slidingWindow_overlap.height  = dynamic_cast<TCLAP::ValueArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3plus.inferenceDeviceType           = o_cmdArguments.useCuda ? torch::kCUDA : torch::kCPU;

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

    // Create and initialize VideoBackgroundSegmentation
    std::unique_ptr<VBGS::VideoBackgroundSegmentation> vbgs(new VBGS::VideoBackgroundSegmentation(cmdArguments.vbgs_settings));
    if(false == vbgs->get_isInitialized()) {
        logging_error("VBGS::VideoBackgroundSegmentation was not correctly initialized");
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
    cv::Mat inputImage, backgroundMask;
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
            logging_info("Image of type " << cv::typeToString(inputImage.type()) << " and size " << inputImage.size());
        }

        // Process background segmentation
        res = vbgs->run(inputImage, backgroundMask);
        if(0 > res) {
            logging_error("VBGS::VideoBackgroundSegmentation::run() failed.");
            return EXIT_FAILURE;
        }

        // Save output
        if(!cmdArguments.outputPath.empty()) {
            logging_info("Saving results in : " << cmdArguments.outputPath);


            std::ostringstream oss;
            oss << cmdArguments.outputPath << "/" << std::setw(8) << std::setfill('0') << cnt++ << ".png";
            std::string path = oss.str();
            logging_info("Writing : " << path);
            cv::imwrite(path, backgroundMask);
        }



        // Draw
        cv::Mat noBackgroundImage = inputImage.clone();
        noBackgroundImage.setTo(cv::Scalar(0, 255, 0), backgroundMask);

        // Display
        cv::imshow("inputImage", inputImage);
        cv::imshow("backgroundMask", backgroundMask);
        cv::imshow("noBackgroundImage", noBackgroundImage);


        logging_info("Push ESC or Q to exit");
        int key = cv::waitKey(1) & 0xff;
        if(27 == key || 'q' == key) {
            break;
        }
    }


    // Manually reset (and delete content of) pointer
    vbgs.reset();

    return res;
}
