
get_filename_component(ProjName ${CMAKE_CURRENT_LIST_DIR} NAME)

file(GLOB SRCS
  "*.cpp"
  )

add_executable(${ProjName}
  ${SRCS}
  )

target_link_libraries(${ProjName}
  tdp 
  ${ISAM_LIBRARY}
  ${CHOLMOD_LIBRARIES}
  #  ${Boost_LIBRARIES}
  )
