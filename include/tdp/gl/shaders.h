#pragma once
#include <pangolin/gl/glsl.h>

namespace tdp {

/// Singelton class to handle loading and compiling of shaders
class Shaders {
 public: 
  static Shaders* Instance();

  pangolin::GlSlProgram labelShader_;
  pangolin::GlSlProgram valueShader_;
  pangolin::GlSlProgram matcapShader_;
 private:
  Shaders();
  ~Shaders() { if (shaders_) delete shaders_; };
  Shaders(const Shaders&) = delete;
  Shaders& operator=(const Shaders&) = delete;
  static Shaders* shaders_;
};

}
