
find_package(Pangolin QUIET)

if (PANGOLIN_FOUND)
  add_subdirectory(./normalsViewer)
endif()
