cmake_minimum_required(VERSION 3.10)
project(MTCNN-MNN)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(
        mnn/include
        src
        src/mtcnn
)

link_directories(mnn/lib)

add_executable(MTCNN-MNN src/main.cpp src/face_detect.cpp src/face_aligner.cpp src/mobileface.cpp)
target_link_libraries(MTCNN-MNN MNN ${OpenCV_LIBS})
