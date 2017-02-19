#pragma once

#include <pangolin/gl/gl.hpp>
#include <tdp/gl/shaders.h>
#include <tdp/gl/labels.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>
#include <tdp/camera/camera_base.h>

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
//  std::cout << camParams.transpose() << " " << w << " " 
//    << h << " " << dMin << " " << dMax << std::endl;
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
  pangolin::GlBuffer& nbo,
  const SE3f& T_cw,
  const CameraBase<float,D,Derived>& cam,
  uint32_t w, uint32_t h,
  float dMin, float dMax, 
  uint32_t numElems) {
  pangolin::GlSlProgram& shader = tdp::Shaders::Instance()->colorByIdOwnCamNormalsShader_;
  shader.Bind();
  Eigen::Vector4f camParams = cam.params_.topRows(4);
//  std::cout << camParams.transpose() << " " << w << " " 
//    << h << " " << dMin << " " << dMax << std::endl;
  shader.SetUniform("cam", camParams(0), camParams(1), camParams(2), camParams(3));
  shader.SetUniform("T_cw", T_cw.matrix());
  shader.SetUniform("w", (float)w);
  shader.SetUniform("h", (float)h);
  shader.SetUniform("dMin", dMin);
  shader.SetUniform("dMax", dMax);
  vbo.Bind();
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
  glEnableVertexAttribArray(0);
  nbo.Bind();
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0); 
  glEnableVertexAttribArray(1);                                               
  glDrawArrays(GL_POINTS, 0, numElems);
  shader.Unbind();
  glDisableVertexAttribArray(1);
  nbo.Unbind();
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
  const pangolin::OpenGlRenderState& cam);

void RenderLabeledVbo(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& labelbo,
  const pangolin::OpenGlRenderState& cam);

void RenderVboIbo(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& ibo);

void RenderVboIboCbo(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& ibo,
  pangolin::GlBuffer& cbo);

void RenderSurfels(
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& nbo,
  pangolin::GlBuffer& cbo,
  pangolin::GlBuffer& rbo,
  float dMax,
  const pangolin::OpenGlMatrix& P,
  const pangolin::OpenGlMatrix& MV);

void RenderVboValuebo(
    const pangolin::GlBuffer& vbo,
    const pangolin::GlBuffer& valuebo,
    float minVal, float maxVal,
    const pangolin::OpenGlMatrix& P,
    const pangolin::OpenGlMatrix& MV
    );

}
