#添加头文件
set ( SRC useGeometry.cpp)

include_directories("/usr/include/eigen3")
add_executable ( test ${SRC} )
