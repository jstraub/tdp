
file(GLOB SRCS "*.cpp" "./*/*cpp")
file(GLOB CU_SRCS "*.cu" "./*/*cu")

if(CUDA_FOUND)
  cuda_add_library(tdp SHARED
    ${SRCS} ${CU_SRCS}
    )
else() 
  add_library(tdp SHARED
    ${SRCS}
    )
endif()
message ("${SRCS}")

target_link_libraries(tdp pangolin)
