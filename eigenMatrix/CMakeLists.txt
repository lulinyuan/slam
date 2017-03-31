#添加头文件
set (SRC eigenMatrix.cpp)

include_directories("/usr/include/eigen3")
add_executable(test ${SRC})
