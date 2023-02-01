cmake_minimum_required(VERSION 3.23)
project(RadRayCu C)

set(CMAKE_C_STANDARD 99)

add_executable(RadRayC main.c radray.c 3dmisc.c)
