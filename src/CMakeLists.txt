
file(GLOB SRCS "*.cpp")
file(GLOB CU_SRCS "*.cu")

if(CUDA_FOUND)
  list(APPEND SRCS ${CU_SRCS})
  cuda_add_library(tdp SHARED
    ${SRCS}
    )
else() 
  add_library(tdp SHARED
    ${SRCS}
    )
endif()
message ("${SRCS}")

target_link_libraries(tdp pangolin)
