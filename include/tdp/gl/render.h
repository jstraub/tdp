#pragma once

#include <pangolin/gl/gl.hpp>
#include <tdp/gl/shaders.h>

namespace tdp {

void RenderLabeledVbo(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& labelbo,
  const pangolin::OpenGlRenderState& cam,
  float maxVal
    ) {
  pangolin::GlSlProgram& shader = tdp::Shaders::Instance()->labelShader_;
  shader.Bind();
  shader.SetUniform("P",cam.GetProjectionMatrix());
  shader.SetUniform("MV",cam.GetModelViewMatrix());
  shader.SetUniform("minValue", 0);
  shader.SetUniform("maxValue", maxVal);
  labelbo.Bind();
  glVertexAttribPointer(1, 1, GL_UNSIGNED_SHORT, GL_FALSE, 0, 0); 
  vbo.Bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
  glEnableVertexAttribArray(0);                                               
  glEnableVertexAttribArray(1);                                               
  //pangolin::RenderVbo(nboA);
  glDrawArrays(GL_POINTS, 0, vbo.num_elements);
  shader.Unbind();
  glDisableVertexAttribArray(1);
  labelbo.Unbind();
  glDisableVertexAttribArray(0);
  vbo.Unbind();
}

}
