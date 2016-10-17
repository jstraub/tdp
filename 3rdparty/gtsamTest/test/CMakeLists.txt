
find_package(GTest REQUIRED)

if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})

  if (GTSAM_FOUND)
    add_executable(testKfSLAM keyframe_slam.cpp)
    target_link_libraries(testKfSLAM ${GTEST_BOTH_LIBRARIES} pthread
      ${GTSAM_LIBS}
      ${Boost_LIBRARIES}
      )
  endif()

endif()
