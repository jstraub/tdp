#pragma once
#include <pangolin/gl/glpixformat.h>

namespace tdp {

template <typename T>
pangolin::GlPixFormat PixFormatFromType(void);

template <typename T>
pangolin::GlPixFormat PixFormatFromType(void) { 
  pangolin::GlPixFormat fmt;
  std::cerr << "unrecognized format for PixFormatFromType" << std::endl;
  return fmt;
}
template <>
pangolin::GlPixFormat PixFormatFromType<uint16_t>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_LUMINANCE;
  fmt.gltype = GL_UNSIGNED_SHORT;
  fmt.scalable_internal_format = GL_LUMINANCE32F_ARB;
  return fmt;
}
template <>
pangolin::GlPixFormat PixFormatFromType<float>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_LUMINANCE;
  fmt.gltype = GL_FLOAT;
  fmt.scalable_internal_format = GL_LUMINANCE32F_ARB;
  return fmt;
}
template <>
pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<float,3,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGB;
  fmt.gltype = GL_FLOAT;
  fmt.scalable_internal_format = GL_RGBA32F;
  return fmt;
}
template <>
pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<float,4,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGBA;
  fmt.gltype = GL_FLOAT;
  fmt.scalable_internal_format = GL_RGBA32F;
  return fmt;
}
template <>
pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<uint8_t,3,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGB;
  fmt.gltype = GL_UNSIGNED_BYTE;
  fmt.scalable_internal_format = GL_RGBA8;
  return fmt;
}
template <>
pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<uint8_t,4,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGBA;
  fmt.gltype = GL_UNSIGNED_BYTE;
  fmt.scalable_internal_format = GL_RGBA8;
  return fmt;
}
}
