project(tdp)
cmake_minimum_required(VERSION 2.4.6)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_COLOR_MAKEFILE ON)
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
ENDIF()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-deprecated -std=c++11")

find_package (CUDA QUIET)
if (CUDA_FOUND)
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35;-std=c++11;-O2;-DVERBOSE")
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  add_definitions(-DCUDA_FOUND)
endif()


find_package(Pangolin REQUIRED)
message(${Pangolin_DIR}/include)
find_package (Eigen3 REQUIRED)

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
