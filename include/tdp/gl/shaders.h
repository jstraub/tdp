#pragma once
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/gl.h>

namespace tdp {

/// Singelton class to handle loading and compiling of shaders
class Shaders {
 public: 
  static Shaders* Instance();

  pangolin::GlSlProgram normalMeshShader_;
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

void RenderVboValuebo(
    const pangolin::GlBuffer& vbo,
    const pangolin::GlBuffer& valuebo,
    float minVal, float maxVal,
    const pangolin::OpenGlMatrix& P,
    const pangolin::OpenGlMatrix& MV
    ) {
  pangolin::GlSlProgram& shader = Shaders::Instance()->valueShader_;

  shader.Bind();
  shader.SetUniform("P", P);
  shader.SetUniform("MV",MV);
  shader.SetUniform("minValue", minVal);
  shader.SetUniform("maxValue", maxVal);

  vbo.Bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
  valuebo.Bind();
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0); 

  glEnableVertexAttribArray(0);                                               
  glEnableVertexAttribArray(1);                                               

  glDrawArrays(GL_POINTS, 0, vbo.num_elements);

  shader.Unbind();
  glDisableVertexAttribArray(1);
  valuebo.Unbind();
  glDisableVertexAttribArray(0);
  vbo.Unbind();

}

}
