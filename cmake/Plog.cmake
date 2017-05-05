include(ExternalProject)
ExternalProject_Add(
  Plog

  # Download Steps
  GIT_REPOSITORY https://github.com/SergiusTheBest/plog.git
  GIT_TAG        master

  # Update Steps
  UPDATE_COMMAND ${GIT_EXECUTABLE} pull

  # Config Steps
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/plog"
  BINARY_DIR ""
  CONFIGURE_COMMAND ""

  # Build Steps
  BUILD_COMMAND ""

  # Install Steps
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(Plog source_dir)
set(Plog_FOUND true)
set(Plog_INCLUDE_DIR "${source_dir}/include")
include_directories(${Plog_INCLUDE_DIR})

message("Included Plog")

