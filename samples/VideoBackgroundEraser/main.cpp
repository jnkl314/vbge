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
    std::string outputPathGrid;

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
                                                                                          "Path to a directory to save rgba result",
                                                                                          false, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("p", "outputPathGrid",
                                                                                          "Path to a directory to save rgb result with grid",
                                                                                          false, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("m", "DeepLabV3ModelPath",
                                                                                          "Path to a PyTorch JIT binary .pb containing the trained model DeepLabV3",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<std::string>("n", "DeepImageMattingModelPath",
                                                                                          "Path to a PyTorch JIT binary .pb containing the trained model DeepImageMatting",
                                                                                          true, "", "string", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::MultiArg<int>       ("b", "background_classId_list",
                                                                                         "IDs of the background in the model",
                                                                                         false, "list<int>", cmd)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::SwitchArg           ("t", "enable_temporalManagement",
                                                                                        "Enable temporal management of scene to improve accuracy between frames. Might not work well for video where the background is moving",
                                                                                        cmd, false)));
        tclap_args.push_back(std::shared_ptr<TCLAP::Arg>(new TCLAP::ValueArg<float>     ("r", "imageMatting_scale",
                                                                                         "Rescale for Deep Image Matting",
                                                                                         false, 1.f, "float", cmd)));




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
    o_cmdArguments.inputPath      = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.useCuda        = dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue();
    o_cmdArguments.hideDisplay    = dynamic_cast<TCLAP::SwitchArg*>            (tclap_args[idx++].get())->getValue();
    o_cmdArguments.outputPath     = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    o_cmdArguments.outputPathGrid = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();

    auto& deeplabv3 = o_cmdArguments.vbge_settings.deeplabv3_inference;
    auto& deepimagematting = o_cmdArguments.vbge_settings.deepimagematting_inference;
    deeplabv3.model_path                    = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    deepimagematting.model_path             = dynamic_cast<TCLAP::ValueArg<std::string>*>(tclap_args[idx++].get())->getValue();
    auto& classId_vector = dynamic_cast<TCLAP::MultiArg<int>*>        (tclap_args[idx++].get())->getValue();
    deeplabv3.background_classId_vector     = classId_vector.empty() ? (std::vector<int>() = {0}) : classId_vector;
    deeplabv3.inferenceDeviceType           = o_cmdArguments.useCuda ? torch::kCUDA : torch::kCPU;
    deepimagematting.inferenceDeviceType    = deepimagematting.inferenceDeviceType;

    o_cmdArguments.vbge_settings.enable_temporalManagement = dynamic_cast<TCLAP::SwitchArg*>      (tclap_args[idx++].get())->getValue();
    o_cmdArguments.vbge_settings.imageMatting_scale        = dynamic_cast<TCLAP::ValueArg<float>*>(tclap_args[idx++].get())->getValue();

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

    // Prepare grid background
    cv::Mat gridBackground;
    {
        gridBackground.create(18, 32, CV_8UC3);
        for(int y = 0 ; y < gridBackground.rows ; ++y) {
            for(int x = 0, b = y%2 ; x < gridBackground.cols ; ++x, b=!b) {
                gridBackground.at<cv::Vec3b>(y, x) = cv::Vec3b::all(b ? 255 : 128);
            }
        }
    }

    // Main loop
    cv::Mat inputImage_bgr, inputImage_rgb;
    cv::Mat outputImage_bgra, outputImage_rgba;
    cv::Mat gridOutputImage_bgr, gridOutputImage_rgb;
    int cnt = 0;
    while(true)
    {
        // Load image
        logging_info("Grab next image, cnt = " << cnt++);
        vc >> inputImage_bgr;
        if(inputImage_bgr.empty()) {
            logging_error("Failed to grab new image");
            break;
        } else {
            logging_info("Image of type " << cv::typeToString(inputImage_bgr.type()) << " and size " << inputImage_bgr.size());
            if(CV_8UC3 != inputImage_bgr.type()) {
                logging_error("CV_8UC3 != inputImage_bgr.type()");
                return EXIT_FAILURE;
            }
        }

        // Convert input image from BGR to RGB
        cv::cvtColor(inputImage_bgr, inputImage_rgb, cv::COLOR_BGR2RGB);

        //-- Main method
        // Process background segmentation and removal
        res = vbge->run(inputImage_rgb, outputImage_rgba);
        if(0 > res) {
            logging_error("VBGE::VideoBackgroundEraser::run() failed.");
            return EXIT_FAILURE;
        }

        // Create grid output image
        if(outputImage_rgba.size() != gridBackground.size()) {
            cv::resize(gridBackground, gridBackground, outputImage_rgba.size(), 0, 0, cv::INTER_NEAREST);
        }
        gridOutputImage_rgb.create(outputImage_rgba.size(), CV_8UC3);
        for(int y = 0, b = 0 ; y < gridOutputImage_rgb.rows ; ++y) {
            for(int x = 0 ; x < gridOutputImage_rgb.cols ; ++x, b^=1) {
                cv::Vec4b &out_im = outputImage_rgba.at<cv::Vec4b>(y, x);
                cv::Vec3b &grid = gridBackground.at<cv::Vec3b>(y, x);
                cv::Vec3b &grid_im = gridOutputImage_rgb.at<cv::Vec3b>(y, x);
                const float alpha = out_im(3)/255.f;
                for(int c = 0 ; c < 3 ; ++c) {
                    grid_im(c) = alpha * out_im(c) + (1.f - alpha) * grid(c);
                }
            }
        }

        // Convert gridOutputImage from RGBA to BGRA
        cv::cvtColor(gridOutputImage_rgb, gridOutputImage_bgr, cv::COLOR_RGB2BGR);
        // Convert gridOutputImage from RGBA to BGRA
        cv::cvtColor(outputImage_rgba, outputImage_bgra, cv::COLOR_RGBA2BGRA);

        // Lambda function to save image
        auto save_function = [cnt](const std::string& i_directory_path, const cv::Mat& i_image) -> bool {
            logging_info("Saving results in : " << i_directory_path);
            std::ostringstream oss;
            oss << i_directory_path << "/" << std::setw(8) << std::setfill('0') << cnt << ".png";
            std::string path = oss.str();
            logging_info("Writing : " << path);
            return cv::imwrite(path, i_image);
        };

        // Save rgba output
        if(!cmdArguments.outputPath.empty() && !save_function(cmdArguments.outputPath, outputImage_bgra)) {
            logging_error("Failed to write outputImage_bgra in :" << cmdArguments.outputPath);
            return EXIT_FAILURE;
        }

        // Save output with grid background
        if(!cmdArguments.outputPathGrid.empty() && !save_function(cmdArguments.outputPathGrid, gridOutputImage_bgr)) {
            logging_error("Failed to write gridOutputImage_bgr in :" << cmdArguments.outputPathGrid);
            return EXIT_FAILURE;
        }

        // Display
        if(false == cmdArguments.hideDisplay) {
            cv::imshow("inputImage", inputImage_bgr);
            cv::imshow("outputImage", outputImage_rgba);
            cv::imshow("gridOutputImage", gridOutputImage_rgb);
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
