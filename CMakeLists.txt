project(tdp)
cmake_minimum_required(VERSION 2.4.6)

if(POLICY CMP0012)
  cmake_policy(SET CMP0012 OLD)
endif()


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

# No OpenMP since it messes with GTSAM
find_package (OpenMP QUIET)
if (OPENMP_FOUND) 
  message("found OpenMP")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
endif()

find_package (CUDA QUIET)
if (CUDA_FOUND)

  set(CUDA_PROPAGATE_HOST_FLAGS FALSE)

  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35;-O2;-DVERBOSE;-D__CUDACC__;")

  set (CUDA_DEBUG 0)
  if (CUDA_DEBUG)
    list(APPEND CUDA_NVCC_FLAGS " -g -G ")
  endif()

  include_directories(${CUDA_TOOLKIT_INCLUDE})
  #add_definitions(-DCUDA_FOUND)
  if (${CUDA_VERSION} LESS 7.0)
    message("CUDA below 7.0")
    list(APPEND CUDA_NVCC_FLAGS ";-std=c++11; -relaxed-constexpr")
  elseif (${CUDA_VERSION} LESS 8.0)
    message("CUDA below 8.0")
    list(APPEND CUDA_NVCC_FLAGS ";-std=c++11; -expt-relaxed-constexpr")
  else()
    list(APPEND CUDA_NVCC_FLAGS ";-std=c++11; -gencode arch=compute_61,code=sm_61; -expt-relaxed-constexpr")
  endif()
endif()

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

option(BUILD_BB "Build Branch&Bound" OFF)

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
endif()

#find_package (OpenNI2 QUIET)
#if (OPENNI2_FOUND) 
#  message("found OpenNI2")
#  include_directories(${OPENNI2_INCLUDE_DIRS})
#  link_directories(${OPENNI2_LIB_DIR})
#endif()

find_package (Calibu QUIET)
if (Calibu_FOUND) 
  message("found Calibu")
  include_directories(${Calibu_INCLUDE_DIRS})
endif()

find_package (ASIO QUIET)
if (ASIO_FOUND) 
  message("found ASIO")
  include_directories(${ASIO_INCLUDE_DIRS})
endif()

find_package (ANN QUIET)
if (ANN_FOUND) 
  message("found ANN")
  include_directories(${ANN_INCLUDE_DIR})
endif()


# Help GTSAM be found
set(GTSAM_DIR "/home/jstraub/workspace/research/3rdparty/gtsam/")
find_package (GTSAM QUIET)
if (GTSAM_FOUND) 
  message("found GTSAM")
  message(${GTSAM_LIBS})
  set(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu")
  find_package(Boost REQUIRED serialization system filesystem thread program_options date_time timer chrono)

  include_directories(
    ${GTSAM_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    )
endif()

set(LIB_SEARCH_DIR "/home/jstraub/workspace/research/3rdparty/isam/")
find_package (ISAM QUIET)
if (ISAM_FOUND) 
  message("found ISAM")
  find_package(Cholmod REQUIRED)

  include_directories(
    ${ISAM_INCLUDE_DIR}
    ${CHOLMOD_INCLUDES}
    )
endif()


include_directories(
  include
  ./3rdparty/fast-C-src/
  ./3rdparty/spectra/include/
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_DIR}/include
  ${Pangolin_DIR}/../../include
  ${CMAKE_BINARY_DIR}/include
)

link_directories(
  /usr/lib/
  ${Pangolin_DIR}
)

set(TDP_LIBS "tdp")

add_subdirectory(./test)
add_subdirectory(./experiments)
add_subdirectory(./src)

set(SHADER_DIR "${PROJECT_SOURCE_DIR}/shaders/")
set(CONFIG_DIR "${PROJECT_SOURCE_DIR}/config/")

configure_file(./include/tdp/config.h.in ./include/tdp/config.h)

