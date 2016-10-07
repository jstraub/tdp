
get_filename_component(ProjName ${CMAKE_CURRENT_LIST_DIR} NAME)

file(GLOB SRCS
  "*.cpp"
  )

link_directories(${aruco_LIB_DIR})

add_executable(${ProjName}
  ${SRCS}
  )
target_link_libraries(${ProjName}
  tdp
  ${aruco_LIBS}
  )
