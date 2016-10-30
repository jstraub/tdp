
get_filename_component(ProjName ${CMAKE_CURRENT_LIST_DIR} NAME)

file(GLOB SRCS
  "*.cpp"
  "../../3rdparty/fast-C-src/*c"
  )

add_executable(${ProjName}
  ${SRCS}
  )
target_link_libraries(${ProjName}
  tdp)
