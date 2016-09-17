project(tdp)
cmake_minimum_required(VERSION 2.4.6)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_COLOR_MAKEFILE ON)
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
ENDIF()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH} "/home/jstraub/workspace/research/3rdparty/install/lib/cmake/")
message(${CMAKE_MODULE_PATH})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG -std=c++11 ")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-deprecated -std=c++11")

#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package (CUDA QUIET)
if (CUDA_FOUND)
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35;-O2;-DVERBOSE;-D__CUDACC__")
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  add_definitions(-DCUDA_FOUND)
endif()

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

find_package(Pangolin REQUIRED)
message(${Pangolin_DIR}/include)

#TODO fogure out hoew to get FindEigen3.cmake to find my own install (need CUDA support)
#unset(EIGEN3_INCLUDE_DIR CACHE)
find_package (Eigen3 3.2.93 REQUIRED)
#set(EIGEN3_INCLUDE_DIR "/home/jstraub/workspace/research/3rdparty/eigen/")
message(${EIGEN3_INCLUDE_DIR})

find_package(aruco 2.0.9 QUIET)
if (aruco_FOUND) 
  message("found aruco")
  include_directories(${aruco_INCLUDE_DIRS})
  link_directories(${aruco_LIB_DIR})
endif()

find_package (OpenNI2 QUIET)
if (OPENNI2_FOUND) 
  message("found OpenNI2")
  include_directories(${OPENNI2_INCLUDE_DIRS})
  link_directories(${OPENNI2_LIB_DIR})
endif()

find_package (Calibu QUIET)
if (Calibu_FOUND) 
  message("found Calibu")
  include_directories(${Calibu_INCLUDE_DIRS})
  link_directories(${Calibu_LIBRARY_DIRS})
endif()

find_package (ASIO QUIET)
if (ASIO_FOUND) 
  message("found ASIO")
  include_directories(${ASIO_INCLUDE_DIRS})
endif()

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_DIR}/include
  ${Pangolin_DIR}/../../include
)

link_directories(
  /usr/lib/
  ${Pangolin_DIR}
)

add_subdirectory(./test)
add_subdirectory(./experiments)
add_subdirectory(./src)
