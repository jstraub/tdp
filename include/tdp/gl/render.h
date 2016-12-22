#pragma once

#include <pangolin/gl/gl.hpp>
#include <tdp/gl/shaders.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

template<int D, typename Derived>
void RenderVboIds(
  pangolin::GlBuffer& vbo,
  const SE3f& T_cw,
  const CameraBase<float,D,Derived>& cam,
  uint32_t w, uint32_t h,
  float dMin, float dMax, 
  uint32_t numElems) {
  pangolin::GlSlProgram& shader = tdp::Shaders::Instance()->colorByIdOwnCamShader_;
  shader.Bind();
  Eigen::Vector4f camParams = cam.params_.topRows(4);
  std::cout << camParams.transpose() << " " << w << " " 
    << h << " " << dMin << " " << dMax << std::endl;
  shader.SetUniform("cam", camParams(0), camParams(1), camParams(2), camParams(3));
  shader.SetUniform("T_cw", T_cw.matrix());
  shader.SetUniform("w", (float)w);
  shader.SetUniform("h", (float)h);
  shader.SetUniform("dMin", dMin);
  shader.SetUniform("dMax", dMax);
  vbo.Bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
  glEnableVertexAttribArray(0);                                               
  glDrawArrays(GL_POINTS, 0, numElems);
  shader.Unbind();
  glDisableVertexAttribArray(0);
  vbo.Unbind();
}

template<int D, typename Derived>
void RenderVboIds(
  pangolin::GlBuffer& vbo,
  const SE3f& T_cw,
  const CameraBase<float,D,Derived>& cam,
  uint32_t w, uint32_t h,
  float dMin, float dMax) {
  return RenderVboIds( vbo, T_cw, cam, w, h, dMin, dMax, vbo.num_elements);
}

void RenderVboIds(
  pangolin::GlBuffer& vbo,
  const pangolin::OpenGlRenderState& cam
  ) {
  pangolin::GlSlProgram& shader = tdp::Shaders::Instance()->colorByIdShader_;
  shader.Bind();
  shader.SetUniform("P",cam.GetProjectionMatrix());
  shader.SetUniform("MV",cam.GetModelViewMatrix());
  vbo.Bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
  glEnableVertexAttribArray(0);                                               
  glDrawArrays(GL_POINTS, 0, vbo.num_elements);
  shader.Unbind();
  glDisableVertexAttribArray(0);
  vbo.Unbind();
}

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

void RenderVboIbo(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& ibo
    ) {
	vbo.Bind();
	glVertexPointer(vbo.count_per_element, vbo.datatype, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	ibo.Bind();
	glDrawElements(GL_TRIANGLES,ibo.num_elements*3, ibo.datatype, 0);
	ibo.Unbind();

	glDisableClientState(GL_VERTEX_ARRAY);
	vbo.Unbind();
}

void RenderVboIboCbo(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& ibo,
  pangolin::GlBuffer& cbo
    ) {
	cbo.Bind();
	glColorPointer(cbo.count_per_element, cbo.datatype, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	tdp::RenderVboIbo(vbo,ibo);

	glDisableClientState(GL_COLOR_ARRAY);
	cbo.Unbind();
}

}
