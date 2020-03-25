CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
project(VideoBackgroundEraser)

######################################
########### CMake Options ############
######################################
set(OPENCV_VERSION "" CACHE STRING "OpenCV Version to specify")

######################################
######### Add Torch Library ##########
######################################
# For some reason, there is a conflict when calling twice "find_package(Torch REQUIRED)"
# in the same project (i.e. from "code" and "sample", or from two dependencies)
# To bypass this behaviour, the line "find_package(Torch REQUIRED)"
# must be called only from the main CMakeLists.txt
if(NOT TORCH_LIBRARIES)
    if(NOT Torch_DIR)
        message("Torch_DIR was not set, using default location : /usr/local/libtorch/share/cmake/Torch")
        set(Torch_DIR /usr/local/libtorch/share/cmake/Torch)
    endif()
    find_package(Torch REQUIRED)
endif()

######################################
########### Create target ############
######################################
# Create Exe
add_executable(${PROJECT_NAME} main.cpp)

######################################
############ Add modules  ############
######################################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../modules ${CMAKE_BINARY_DIR}/modules)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/include)
target_link_libraries(${PROJECT_NAME} VBGE_modules)

######################################
######### Add OpenCV Library #########
######################################
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

######################################
########### Build Options ############
######################################
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic -Wall -pthread")

