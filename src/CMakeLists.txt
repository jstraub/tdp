
file(GLOB SRCS 
  "./cuda/*cpp"
  "./directional/*cpp"
  "./gui/*cpp"
  "./icp/*cpp"
  "./manifold/*cpp"
  "./plane/*cpp"
  "./preproc/*cpp"
  "./reductions/*cpp"
  "./rtmf/*cpp"
  "./stats/*cpp"
  "./tsdf/*cpp"
  "./utils/*cpp"
  "./inertial/*cpp"
  "./drivers/inertial/imu_pango.cpp"
  "./io/*cpp"
  "./bb/*cpp"
  "./gl/*cpp"
  )
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

if (Calibu_FOUND) 
  list(APPEND LIBS ${Calibu_LIBRARIES})
endif()

if (aruco_FOUND) 
  list(APPEND LIBS ${aruco_LIBS})
  list(APPEND SRCS "./marker/aruco.cpp")
endif()

target_link_libraries(tdp pangolin
  ${LIBS}
  )
