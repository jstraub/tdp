
find_package(GTest REQUIRED)

if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})

#  add_executable(s3 S3.cpp)
#  target_link_libraries(s3 tdp ${GTEST_BOTH_LIBRARIES} pthread)
#
add_executable(SE3 SE3.cpp)
target_link_libraries(SE3 tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(SE3derivs SE3derivs.cpp)
target_link_libraries(SE3derivs tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(SO3derivs SO3derivs.cpp)
target_link_libraries(SO3derivs tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(SO3 SO3.cpp)
target_link_libraries(SO3 tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(spectra spectra.cpp)
target_link_libraries(spectra ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(image image.cpp)
target_link_libraries(image tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(ransac ransac.cpp)
target_link_libraries(ransac tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(lsh lsh.cpp)
target_link_libraries(lsh tdp ${GTEST_BOTH_LIBRARIES} pthread)


add_executable(cublas cublas.cpp)
target_link_libraries(cublas tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(vmfmmf vmfmmf.cpp)
target_link_libraries(vmfmmf tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(testLaplace_beltrami laplace_beltrami.cpp)
target_link_libraries(testLaplace_beltrami tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(testEigen_io eigen_io.cpp)
target_link_libraries(testEigen_io tdp ${GTEST_BOTH_LIBRARIES} pthread)

add_executable(cg cg.cpp)
target_link_libraries(cg tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(projective_assoc projective_assoc.cpp)
  target_link_libraries(projective_assoc tdp ${GTEST_BOTH_LIBRARIES} pthread)

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

  add_executable(testNN_Cuda nn_cuda.cpp)
  target_link_libraries(testNN_Cuda tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testParallelSort parallelSort.cpp)
  target_link_libraries(testParallelSort tdp ${GTEST_BOTH_LIBRARIES} pthread)

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
