include(ExternalProject)
ExternalProject_Add(
  Spectra

  # Download Steps
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        master

  # Update Steps
  UPDATE_COMMAND ${GIT_EXECUTABLE} pull

  # Config Steps
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/spectra"
  BINARY_DIR ""
  CONFIGURE_COMMAND ""

  # Build Steps
  BUILD_COMMAND ""

  # Install Steps
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(Spectra source_dir)
set(Spectra_FOUND true)
set(Spectra_INCLUDE_DIR "${source_dir}/include")
# Note that this project does not build into a library

include_directories(${Spectra_INCLUDE_DIR})

message("Included Spectra")

