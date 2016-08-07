
find_package(GTest REQUIRED)

if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIRS})

  add_executable(testCamera camera.cpp)
  target_link_libraries(testCamera 3dp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testImage image.cpp)
  target_link_libraries(testImage 3dp ${GTEST_BOTH_LIBRARIES} pthread)

  add_executable(testVolume volume.cpp)
  target_link_libraries(testVolume 3dp ${GTEST_BOTH_LIBRARIES} pthread)
endif()
