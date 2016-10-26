
find_package(GTest REQUIRED)

if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})

#  add_executable(s3 S3.cpp)
#  target_link_libraries(s3 tdp ${GTEST_BOTH_LIBRARIES} pthread)
#
add_executable(SE3 SE3.cpp)
target_link_libraries(SE3 tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(SO3 SO3.cpp)
target_link_libraries(SO3 tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(image image.cpp)
target_link_libraries(image tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(sl3 SL3.cpp)
  target_link_libraries(sl3 tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testCamera camera.cpp)
  target_link_libraries(testCamera tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testVolume volume.cpp)
  target_link_libraries(testVolume tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testProjective_math projective_math.cpp)
  target_link_libraries(testProjective_math tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testRig rig.cpp)
  target_link_libraries(testRig tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testDirectionalHist directionalHist.cpp)
  target_link_libraries(testDirectionalHist tdp ${GTEST_BOTH_LIBRARIES} pthread)

  if (ANN_FOUND)
    add_executable(testANN ann.cpp)
    target_link_libraries(testANN tdp ${GTEST_BOTH_LIBRARIES} pthread)
  endif()

#  if (GTSAM_FOUND)
#    add_executable(testKfSLAM keyframe_slam.cpp)
#    target_link_libraries(testKfSLAM 
#      tdp 
#      ${GTEST_BOTH_LIBRARIES} 
#      pthread
#      ${GTSAM_LIBS}
#      ${Boost_LIBRARIES}
#      )
#  endif()

  if (ISAM_FOUND)
    add_executable(testISAM isam.cpp)
    target_link_libraries(testISAM
      tdp 
      ${ISAM_LIBRARY}
      ${CHOLMOD_LIBRARIES}
      ${GTEST_BOTH_LIBRARIES} 
      pthread
      ${ISAM_LIBRARIES}
      )
  endif()



endif()
