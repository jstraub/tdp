include(ExternalProject)
ExternalProject_Add(
  ANN

  # Download Steps
  DOWNLOAD_COMMAND ""

  # Update Steps
  UPDATE_COMMAND ""

  # Config Steps
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ann_1.1.2"
  BINARY_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ann_1.1.2"
  CONFIGURE_COMMAND ""

  # Build Steps
  BUILD_COMMAND mkdir -p lib && make linux-g++-notest

  # Install Steps
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(ANN source_dir)
set(ANN_FOUND true)
set(ANN_INCLUDE_DIR "${source_dir}/include")
set(ANN_LIBRARIES "${source_dir}/lib/${CMAKE_SHARED_MODULE_PREFIX}ANN${CMAKE_SHARED_LIBRARY_SUFFIX}")
include_directories(${ANN_INCLUDE_DIR})

message("Included ANN")

