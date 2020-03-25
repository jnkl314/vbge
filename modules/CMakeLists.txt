CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
project(VBGE_modules)
set(TARGET_NAME ${PROJECT_NAME})

######################################
########### CMake Options ############
######################################
option(VBGS_WITH_VERBOSE "Enable verbose mode" ON)

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
# List source files
file(GLOB_RECURSE LOCAL_SOURCES src/*.cpp)
# List header files
file(GLOB_RECURSE LOCAL_HEADERS include/*.hpp private_include/*.hpp)
# Create Lib
add_library(${PROJECT_NAME} ${LOCAL_SOURCES} ${LOCAL_HEADERS})
target_include_directories(${TARGET_NAME}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private_include)

######################################
###### Add Torch Library Part 2 ######
######################################
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

######################################
######### Add OpenCV Library #########
######################################
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

######################################
########### Build Options ############
######################################
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpic -Wall")

######################################
########## Add Definitions ###########
######################################
if(VBGS_WITH_VERBOSE)
    target_compile_definitions(${TARGET_NAME} PUBLIC VBGS_ENABLE_VERBOSE)
endif(VBGS_WITH_VERBOSE)
