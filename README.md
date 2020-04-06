# Video Background Eraser

## Result Preview
The pipeline produces RGBA images.<br/>
Background pixels have alpha = 0.<br/>
The contour of the foreground objects have progressive alpha values to blend them with new backgrounds.<br/>
<img src="./pictures/A_image_original.png" width="432" height="243"><img src="./pictures/A_image_withoutBackground.png" width="432" height="243"><br/>
<img src="./pictures/B_image_original.png" width="432" height="243"><img src="./pictures/B_image_withoutBackground.png" width="432" height="243"><br/>
<img src="./pictures/mix_original_newBG.gif">

## Requirements
Tested on Ubuntu 18.04.

Direct dependencies: <br/>
* TCLAP
  * sudo apt install libtclap-dev
* libtorch
  * (Cuda enabled) https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.4.0.zip
  * (without Cuda) https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip <br/>
* Opencv 4.2.0
  * built from sources (https://github.com/opencv/opencv/tree/4.2.0 and contribs https://github.com/opencv/opencv_contrib/tree/4.2.0).

## Components
The pipeline is composed by :<br/>
* Semantic Segmentation using DeepLabV3
  * see my other repository https://github.com/jnkl314/DeepLabV3FineTuning
  * weights were fine-tuned for skydiver segmentation, and frozen with PyTorch JIT Tracing
* Optical flow to track pixels of foreground objects over time
* Trimap generation using morpho math
* Deep Image Matting to improve alpha on objects contour
  * pretrained weights come from https://github.com/foamliu/Deep-Image-Matting-PyTorch
  * also frozen using PyTorch JIT Tracing
  
## Build
```bash
cd ./samples/
mkdir build
cd build
cmake ../VideoBackgroundEraser/
make -j8
```

## Usage
```bash
USAGE: 

 VideoBackgroundEraser  [-r <float>] [-t]
                              [--slidingWindow_overlapHeight <int>]
                              [--slidingWindow_overlapWidth <int>]
                              [--slidingWindow_height <int>]
                              [--slidingWindow_width <int>] [-s] [-b
                              <list<int>>] ...  -n <string> -m <string>
                              [-p <string>] [-o <string>] [--hideDisplay]
                              [-c] -i <string> [--] [--version] [-h]
  Where: 

   -r <float>,  --imageMatting_scale <float>
     Rescale for Deep Image Matting

   -t,  --enable_temporalManagement
     Enable temporal management of scene to improve accuracy between
     frames. Might not work well for video where the background is moving

   --slidingWindow_overlapHeight <int>
     Width of the overlap between sliding windows

   --slidingWindow_overlapWidth <int>
     Height of the overlap between sliding windows

   --slidingWindow_height <int>
     Height of the inference for the given model

   --slidingWindow_width <int>
     Width of the inference for the given model

   -s,  --useSlidingWindow
     Instead of resizing the input image to 513x513, use a sliding window
     and merge masks

   -b <list<int>>,  --background_classId_list <list<int>>  (accepted
      multiple times)
     IDs of the background in the model

   -n <string>,  --DeepImageMattingModelPath <string>
     (required)  Path to a PyTorch JIT binary .pb containing the trained
     model DeepImageMatting

   -m <string>,  --DeepLabV3ModelPath <string>
     (required)  Path to a PyTorch JIT binary .pb containing the trained
     model DeepLabV3

   -p <string>,  --outputPathGrid <string>
     Path to a directory to save rgb result with grid

   -o <string>,  --outputPath <string>
     Path to a directory to save rgba result

   --hideDisplay
     Hide display of source image and result

   -c,  --useCuda
     Use Cuda for inference

   -i <string>,  --inputPath <string>
     (required)  Path to video or a directory+pattern

  ```
