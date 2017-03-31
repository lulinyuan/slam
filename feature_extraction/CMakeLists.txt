cmake_minimum_required( VERSION 2.8 )
project( feature_extraction )

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set( CMAKE_BUILD_TYPE "Release" )

#添加opencv库
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})


add_executable( feature_extraction feature_extraction.cpp )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )


