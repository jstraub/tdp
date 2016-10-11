
#include <string>

#include <tdp/config.h>
#include <tdp/gl/shaders.h>

namespace tdp {

Shaders* Shaders::shaders_ = nullptr;

Shaders* Shaders::Instance() {
  if (!shaders_)
    shaders_ = new Shaders;
  return shaders_;
}

Shaders::Shaders() {
  std::string shaderRoot = SHADER_DIR;
  labelShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("labelShading.vert"));
  labelShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("setColor.frag"));
  labelShader_.Link();
}

}
