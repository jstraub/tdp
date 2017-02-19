
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


  // shader to render a point cloud with labels
  labelShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("labelShading.vert"));
  labelShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("labelShading.frag"));
  labelShader_.Link();

  // shader to render a point cloud with a value attached to each
  // point
  valueShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("valueShading.vert"));
  valueShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("setColor.frag"));
  valueShader_.Link();

  matcapShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("matcap.vert"));
  matcapShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("matcap.frag"));
  matcapShader_.Link();

  colorByIdShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("colorById.vert"));
  colorByIdShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("colorById.frag"));
  colorByIdShader_.Link();

  colorByIdOwnCamShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("colorByIdOwnCam.vert"));
  colorByIdOwnCamShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("colorById.frag"));
  colorByIdOwnCamShader_.Link();

  colorByIdOwnCamNormalsShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("colorByIdOwnCamNormals.vert"));
  colorByIdOwnCamNormalsShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("colorById.frag"));
  colorByIdOwnCamNormalsShader_.Link();

  normalMeshShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("normalShading.vert"));
  normalMeshShader_.AddShaderFromFile(pangolin::GlSlGeometryShader, 
      shaderRoot+std::string("normalShading.geom"));
  normalMeshShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("normalShading.frag"));
  normalMeshShader_.Link();

  surfelShader_.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("surfelRgbMVP.vert"));
  surfelShader_.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("surfelRgbMVP.frag"));
  surfelShader_.Link();

}

}
