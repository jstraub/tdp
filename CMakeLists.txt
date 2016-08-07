project(3dp)
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
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")
  include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

find_package (Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

link_directories(
  /usr/lib/
)

add_subdirectory(./test)
add_subdirectory(./src)
