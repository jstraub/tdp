#pragma once
#include <pangolin/gl/glpixformat.h>

namespace tdp {

template <typename T>
inline pangolin::GlPixFormat PixFormatFromType(void) { 
  pangolin::GlPixFormat fmt;
  std::cerr << "unrecognized format for PixFormatFromType " << typeid(T).name() << std::endl;
  return fmt;
}

template <>
inline pangolin::GlPixFormat PixFormatFromType<uint16_t>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_LUMINANCE;
  fmt.gltype = GL_UNSIGNED_SHORT;
  fmt.scalable_internal_format = GL_LUMINANCE32F_ARB;
  //std::cout << "GL_LUMINANCE; GL_UNSIGNED_SHORT; GL_LUMINANCE32F_ARB;" << std::endl;
  return fmt;
}
template <>
inline pangolin::GlPixFormat PixFormatFromType<float>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_LUMINANCE;
  fmt.gltype = GL_FLOAT;
  fmt.scalable_internal_format = GL_LUMINANCE32F_ARB;
  //std::cout << "GL_LUMINANCE; GL_FLOAT; GL_LUMINANCE32F_ARB;" << std::endl;
  return fmt;
}
template <>
inline pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<float,3,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGB;
  fmt.gltype = GL_FLOAT;
  fmt.scalable_internal_format = GL_RGBA32F;
  return fmt;
}
template <>
inline pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<float,4,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGBA;
  fmt.gltype = GL_FLOAT;
  fmt.scalable_internal_format = GL_RGBA32F;
  return fmt;
}
template <>
inline pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<uint8_t,3,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGB;
  fmt.gltype = GL_UNSIGNED_BYTE;
  fmt.scalable_internal_format = GL_RGBA8;
  return fmt;
}
template <>
inline pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<uint8_t,3,1,Eigen::DontAlign>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGB;
  fmt.gltype = GL_UNSIGNED_BYTE;
  fmt.scalable_internal_format = GL_RGBA8;
  return fmt;
}
template <>
inline pangolin::GlPixFormat PixFormatFromType<Eigen::Matrix<uint8_t,4,1>>(void) { 
  pangolin::GlPixFormat fmt;
  fmt.glformat = GL_RGBA;
  fmt.gltype = GL_UNSIGNED_BYTE;
  fmt.scalable_internal_format = GL_RGBA8;
  return fmt;
}

}
