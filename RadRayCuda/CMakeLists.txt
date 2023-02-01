cmake_minimum_required(VERSION 3.23)
project(RadRayCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(RadRayCuda main.cu radray.cu 3dmisc.cu)
add_library(radray.cu 3dmisc.cu)

set_target_properties(RadRayCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
