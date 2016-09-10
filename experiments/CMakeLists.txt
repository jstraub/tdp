
add_subdirectory(./pcViewer)
if (CUDA_FOUND)
  add_subdirectory(./simpleGui)
  add_subdirectory(./marker)
  add_subdirectory(./depthFusion)
  add_subdirectory(./normalsViewer)
  add_subdirectory(./calibration)

  if (aruco_FOUND)
    add_subdirectory(./markerAruco)
  endif()
endif()
