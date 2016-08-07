
file(GLOB SRCS "*.cpp")

if(CUDA_FOUND)
  list(APPEND SRCS
    ./ray.cu
    ./normals.cu
    ./depth.cu
    )
endif()

add_library(3dp SHARED
  ${SRCS}
  )
