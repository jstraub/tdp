
find_package(GTest REQUIRED)

if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})

  add_executable(testCamera camera.cpp)
  target_link_libraries(testCamera tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testVolume volume.cpp)
  target_link_libraries(testVolume tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testProjective_math projective_math.cpp)
  target_link_libraries(testProjective_math tdp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testSO3 SO3.cpp)
  target_link_libraries(testSO3 tdp)

  add_executable(testSE3 SE3.cpp)
  target_link_libraries(testSE3 tdp)

endif()
