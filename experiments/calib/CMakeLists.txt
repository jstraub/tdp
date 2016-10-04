# This file is part of the Calibu Project.
# https://github.com/arpg/Calibu
# It was adapted to compile with tdp by Julian Straub

# git://github.com/stevenlovegrove/Pangolin.git
#find_package( Pangolin 0.1 QUIET )
#include_directories( ${Pangolin_INCLUDE_DIRS} )

# git clone https://ceres-solver.googlesource.com/ceres-solver
list(APPEND SEARCH_HEADERS ${EIGEN3_INCLUDE_DIR}) # Help Ceres find Eigen
find_package( CeresSolver 1.6.0 QUIET )
include_directories( ${CERES_INCLUDES} )

# Check that OPENCV is available
set(NO_OPENCV_GPU TRUE)
find_package( OpenCV QUIET )

find_package(Calibu QUIET)

if(Pangolin_FOUND AND CeresSolver_FOUND AND OpenCV_FOUND)
  message(${CeresSolver_LIBRARIES})
    add_executable( calib main.cpp )
    include_directories( 
      ${CeresSolver_INCLUDE_DIRS} 
      ${Calibu_INCLUDE_DIRS}
      )
    target_link_libraries( calib
        ${CeresSolver_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${Calibu_LIBRARIES}
        tdp)
#    install(TARGETS calibgrid EXPORT CalibuTargets RUNTIME
#            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
else()
    #report what's missing
    set(WARNING_MSG "calib dependencies not met:")
    foreach(dep Pangolin CeresSolver OpenCV)
        if(NOT ${dep}_FOUND)
            set(WARNING_MSG "${WARNING_MSG} ${dep}")
        endif() 
    endforeach()
    set(WARNING_MSG "${WARNING_MSG}. Skipping calib during build.")
    message(WARNING ${WARNING_MSG})
endif()
