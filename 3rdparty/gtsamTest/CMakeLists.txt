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

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

#TODO fogure out hoew to get FindEigen3.cmake to find my own install (need CUDA support)
#unset(EIGEN3_INCLUDE_DIR CACHE)
find_package (Eigen3 3.2.93 REQUIRED)
#set(EIGEN3_INCLUDE_DIR "/home/jstraub/workspace/research/3rdparty/eigen/")
message(${EIGEN3_INCLUDE_DIR})

# Help GTSAM be found
set(GTSAM_DIR "/home/jstraub/workspace/research/3rdparty/gtsam/")
find_package (GTSAM QUIET)
if (GTSAM_FOUND) 
  message("found GTSAM")
  message(${GTSAM_LIBS})
  find_package(Boost REQUIRED serialization system filesystem thread program_options date_time timer chrono)

  include_directories(
    ${GTSAM_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    )
endif()

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

link_directories(
  /usr/lib/
)

add_subdirectory(./test)
