cmake_minimum_required( VERSION 2.8 )
project( triangulation )

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set( CMAKE_BUILD_TYPE "Release" )

#添加opencv库
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})


add_executable( triangulation triangulation_my.cpp )
target_link_libraries( triangulation ${OpenCV_LIBS} )


