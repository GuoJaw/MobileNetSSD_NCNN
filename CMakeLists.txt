

cmake_minimum_required(VERSION 2.8) # cmake verson，指定cmake版本 
add_definitions(-std=c++11 -g)

project (ncnn_test)


set(ncnn_test_INCLUDE_DIRS /home/gjw/ncnn_mobilenet_ssd/include)

include_directories(
	${ncnn_test_INCLUDE_DIRS} 

)

### OpenCV的配置 ###
set (OpenCV_DIR /usr)  
find_package(OpenCV 2 REQUIRED)

add_executable (ncnn_test main.cpp)


set (NCNN_LIBS /home/gjw/ncnn_mobilenet_ssd/libncnn.a)
set (GOMP_LIB /usr/lib/x86_64-linux-gnu/libgomp.so.1)

target_link_libraries( ncnn_test 
	${OpenCV_LIBS} 
	${NCNN_LIBS} 
	${GOMP_LIB}
)


