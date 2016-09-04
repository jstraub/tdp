
add_subdirectory(./pcViewer)
if (CUDA_FOUND)
  add_subdirectory(./depthFusion)
  add_subdirectory(./normalsViewer)
  add_subdirectory(./calibration)
endif()
