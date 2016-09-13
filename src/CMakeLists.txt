
file(GLOB SRCS "*.cpp" "./*/*cpp")
file(GLOB CU_SRCS "*.cu" "./*/*cu")

if (ASIO_FOUND)
  list(APPEND SRCS "./drivers/inertial/3dmgx3_45.cpp")
endif()

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

target_link_libraries(tdp pangolin
  ${aruco_LIBS}
  )
