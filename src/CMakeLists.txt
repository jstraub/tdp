
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
  "./data/*cpp"
  "./clustering/*cpp"
  "./features/*cpp"
  "./stats/*cpp"
  "./tsdf/*cpp"
  "./utils/*cpp"
  "./inertial/*cpp"
  "./marching_cubes/*cpp"
  "./distributions/*cpp"
  "./drivers/inertial/imu_pango.cpp"
  "./io/*cpp"
  "./gl/*cpp"
  "./laplace_beltrami/*cpp"
  "../3rdparty/fast-C-src/*c"
  "./nn_cuda/*cpp"
  "./sorts/*cpp"
  "./filters/*cpp"
  )
file(GLOB CU_SRCS "*.cu" "./*/*cu")

#file(GLOB ISAM_SRCS "./slam/*.cpp")

file(GLOB BB_SRCS
  "./bb/*cpp"
  )

if (BUILD_BB)
  list(APPEND SRCS ${BB_SRCS})
endif()

if (ISAM_FOUND)
#  list(APPEND SRCS ${ISAM_SRCS})
  list(APPEND TDP_LIBS
      ${ISAM_LIBRARY}
      ${CHOLMOD_LIBRARIES}
    )
endif()

if (ASIO_FOUND)
  list(APPEND SRCS "./drivers/inertial/3dmgx3_45.cpp")
endif()

if(CUDA_FOUND)
  cuda_add_library(tdp SHARED
    ${SRCS} ${CU_SRCS}
    )
  list(APPEND LIBS
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    )
else()
  add_library(tdp SHARED
    ${SRCS}
    )
endif()

if (Calibu_FOUND)
  list(APPEND LIBS ${Calibu_LIBRARIES})
endif()

if (ANN_FOUND)
  list(APPEND LIBS ${ANN_LIBRARIES})
endif()

if (aruco_FOUND)
  list(APPEND LIBS ${aruco_LIBS})
  list(APPEND SRCS "./marker/aruco.cpp")
endif()

add_dependencies(tdp ANN)
target_link_libraries(tdp pangolin
  ${LIBS}
  )
cotire(tdp)

install(TARGETS tdp LIBRARY DESTINATION lib)

