cmake_minimum_required( VERSION 2.8 )
project( ceres_curve_fitting )

set( CAMKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#寻找Ceres库并添加它头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting ceres.cpp )
#与Ceres和OpenCV链接
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )
