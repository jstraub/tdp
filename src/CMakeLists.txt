
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

if (Calibu_FOUND) 
  list(APPEND LIBS ${Calibu_LIBRARIES})
endif()

if (aruco_FOUND) 
  list(APPEND LIBS ${aruco_LIBS})
endif()

target_link_libraries(tdp pangolin
  ${LIBS}
  )
