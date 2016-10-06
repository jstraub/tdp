
get_filename_component(ProjName ${CMAKE_CURRENT_LIST_DIR} NAME)

file(GLOB SRCS
  "*.cpp"
  )

add_executable(${ProjName}
  ${SRCS}
  )

message(${ProjName})
message(${Boost_LIBRARIES})

target_link_libraries(${ProjName}
  ${GTSAM_LIBS}
  ${Boost_LIBRARIES}
  tdp)
