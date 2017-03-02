#pragma once
#include <tdp/data/image.h>
#include <tdp/preproc/plane.h>
#include <tdp/gl/render.h>
#include <tdp/camera/camera_base.h>

namespace tdp {

template<int D, typename Derived>
class ProjectiveAssociation {
 public: 
  ProjectiveAssociation(const CameraBase<float,D,Derived>& cam, 
      size_t w, size_t h)
    : w_(w), h_(h), cam_(cam), tex_(w, h, GL_RGBA), 
     render_(w, h, GL_DEPTH_COMPONENT), fbo_(tex_, render_)
  { }
  ~ProjectiveAssociation() {}

  /// vertices, normals, and times
  void Associate(pangolin::GlBuffer& vbo, pangolin::GlBuffer& nbo,
      pangolin::GlBuffer& tbo,
      SE3f T_cw, float dMin, float dMax,
      int32_t tMin,
      uint32_t numElems) {
    fbo_.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glPointSize(1);
    glViewport(0, 0, w_, h_);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    tdp::RenderVboIds(vbo, nbo, tbo, T_cw, cam_, w_, h_, dMin, dMax,
        tMin, numElems);
    fbo_.Unbind();
    glPopAttrib();
    glFinish();
  }


  void Associate(pangolin::GlBuffer& vbo, pangolin::GlBuffer& nbo,
      SE3f T_cw, float dMin, float dMax, uint32_t numElems) {
    fbo_.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glPointSize(1);
    glViewport(0, 0, w_, h_);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    tdp::RenderVboIds(vbo, nbo, T_cw, cam_, w_, h_, dMin, dMax,
        numElems);
    fbo_.Unbind();
    glPopAttrib();
    glFinish();
  }

  
  void Associate(pangolin::GlBuffer& vbo,
      SE3f T_cw, float dMin, float dMax, uint32_t numElems) {
    fbo_.Bind();

//    glViewport(0,0,w_,h_);
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    glOrtho(-0.5, w_-0.5, -0.5, h_-0.5, -1, 1);
//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();

    glPushAttrib(GL_VIEWPORT_BIT);
    glPointSize(1);
    glViewport(0, 0, w_, h_);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    tdp::RenderVboIds(vbo, T_cw, cam_, w_, h_, dMin, dMax, numElems);
    fbo_.Unbind();
    glPopAttrib();
    glFinish();
  }
  
  /// z_i == 0 means not associated; z_i > 0 means associated to z_i-1
  void GetAssoc(tdp::Image<uint32_t>& z) {
    tex_.Download(z.ptr_, GL_RGBA, GL_UNSIGNED_BYTE);
    for (size_t i=0; i<z.Area(); ++i) {
      z[i] &= 0x00FFFFFF; // alpha channel is 255 always
    }
  }
  void GetAssoc(tdp::Image<uint32_t>& z, tdp::Image<uint8_t>& mask) {
    GetAssoc(z);
    for (size_t i=0; i<z.Area(); ++i) {
      mask[i] = z[i]>0? 1: 0; 
    }
  }
  void GetAssoc(tdp::Image<uint32_t>& z, 
      tdp::Image<uint8_t>& mask, std::vector<uint32_t>& ids) {
    GetAssoc(z);
    for (size_t i=0; i<z.Area(); ++i) {
      mask[i] = z[i]>0? 1: 0; 
      if (mask[i])
        ids.push_back(z[i]-1);
    }
  }
  /// adds depth-based occlusion reasoning to filter data associations
  void GetAssocOcclusion(
      const tdp::Image<tdp::Plane>& pl_w,
      const tdp::Image<tdp::Vector3fda>& pc_c,
      const tdp::SE3f& T_cw,
      float occlusionDepthThr,
      tdp::Image<uint32_t>& z, 
      tdp::Image<uint8_t>& mask, std::vector<uint32_t>& ids) {
    GetAssoc(z);
    // TODO could try to use depth of render buffer
    for (size_t i=0; i<z.Area(); ++i) {
      if (z[i]>0) {
        uint32_t id = z[i]-1;
        float d_w_in_c = (T_cw*pl_w[id].p_)(2);
        float d_c = pc_c[i](2);
        if (fabs(d_w_in_c - d_c) < occlusionDepthThr) {
          mask[i] = 255;
          ids.push_back(id);
        } else {
          mask[i] = 0;
        }
      }
    }
  }

  /// adds depth-based occlusion reasoning to filter data associations
  template<int LEVELS>
  void GetAssocOcclusion(
      const tdp::Image<tdp::Plane>& pl_w,
      const tdp::Pyramid<tdp::Vector3fda,LEVELS>& pyrPc,
      const tdp::SE3f& T_cw,
      float occlusionDepthThr,
      float dMin,
      float dMax,
      tdp::Pyramid<uint32_t,LEVELS>& pyrZ, 
      tdp::Pyramid<uint8_t ,LEVELS>& pyrMask, 
      std::vector<std::vector<uint32_t>*>& ids
      ) {
    tdp::Image<tdp::Vector3fda> pc0 = pyrPc.GetConstImage(0);
    tdp::Image<uint32_t> z0 = pyrZ.GetImage(0);
    tdp::Image<uint8_t> mask0 = pyrMask.GetImage(0);
    GetAssocOcclusion(pl_w, pc0, T_cw, occlusionDepthThr, z0, mask0,
        *ids[0]);
    for (size_t lvl=1; lvl < LEVELS; ++lvl) {
      tdp::Image<tdp::Vector3fda> pc0 = pyrPc.GetConstImage(lvl-1);
      tdp::Image<uint32_t> z0 = pyrZ.GetImage(lvl-1);
      tdp::Image<uint8_t> mask0 = pyrMask.GetImage(lvl-1);
      tdp::Image<tdp::Vector3fda> pc1 = pyrPc.GetConstImage(lvl);
      tdp::Image<uint32_t> z1 = pyrZ.GetImage(lvl);
      tdp::Image<uint8_t> mask1 = pyrMask.GetImage(lvl);
      tdp::Vector4ida zs;
      tdp::Vector4fda ds;
      for (size_t v=0; v<pc1.h_; ++v)  {
        for (size_t u=0; u<pc1.w_; ++u)  {
          zs(0) = z0(2*u,2*v);
          zs(1) = z0(2*u,2*v+1);
          zs(2) = z0(2*u+1,2*v);
          zs(3) = z0(2*u+1,2*v+1);
          ds(0) = pc0(2*u,2*v)(2);
          ds(1) = pc0(2*u,2*v+1)(2);
          ds(2) = pc0(2*u+1,2*v)(2);
          ds(3) = pc0(2*u+1,2*v+1)(2);
          ds(0) = ds(0) < dMin || zs(0) == 0 ? 9999. : ds(0);
          ds(1) = ds(1) < dMin || zs(1) == 0 ? 9999. : ds(1);
          ds(2) = ds(2) < dMin || zs(2) == 0 ? 9999. : ds(2);
          ds(3) = ds(3) < dMin || zs(3) == 0 ? 9999. : ds(3);
          int32_t id = 0;
          float minD = ds.minCoeff(&id);
          z1(u,v) = minD > dMax ? 0 : zs(id);
          if (z1(u,v) > 0) {
            mask1(u,v) = 255;
            ids[lvl]->push_back(z1(u,v)-1);
          } else {
            mask1(u,v) = 0;
          }
        }
      }
    }
  }

  /// adds depth-variance-based occlusion reasoning to filter data associations
  void GetAssocOcclusion(
      const tdp::Image<tdp::Plane>& pl_w,
      const tdp::Image<tdp::Matrix3fda>& covs,
      const tdp::Image<tdp::Vector3fda>& pc_c,
      const tdp::Image<tdp::Vector3fda>& ray,
      const tdp::SE3f& T_cw,
      float numSigmaOclusion,
      tdp::Image<uint32_t>& z, 
      tdp::Image<uint8_t>& mask, 
      std::vector<uint32_t>& ids) {
    GetAssoc(z);
    // TODO could try to use depth of render buffer
    for (size_t i=0; i<z.Area(); ++i) {
      if (z[i]>0) {
        uint32_t id = z[i]-1;
        float d_w_in_c = (T_cw*pl_w[id].p_)(2);
        float d_c = pc_c[i](2);
        float threeSigma_d = numSigmaOclusion*sqrtf(ray[i].dot(covs[id]*ray[i]));
        if (fabs(d_w_in_c - d_c) < threeSigma_d) {
          mask[i] = 255;
          ids.push_back(id);
        } else {
          mask[i] = 0;
        }
      }
    }
  }

  /// adds depth and variance-based occlusion reasoning to filter data associations
  template<int LEVELS>
  void GetAssocOcclusion(
      const tdp::Image<tdp::Plane>& pl_w,
      const tdp::Image<tdp::Matrix3fda>& covs,
      const tdp::Pyramid<tdp::Vector3fda,LEVELS>& pyrPc,
      const tdp::Pyramid<tdp::Vector3fda,LEVELS>& pyrRay,
      const tdp::SE3f& T_cw,
      float numSigmaOclusion,
      float dMin,
      float dMax,
      tdp::Pyramid<uint32_t,LEVELS>& pyrZ, 
      tdp::Pyramid<uint8_t ,LEVELS>& pyrMask, 
      std::vector<std::vector<uint32_t>*>& ids
      ) {
    tdp::Image<tdp::Vector3fda> pc0 = pyrPc.GetConstImage(0);
    tdp::Image<tdp::Vector3fda> ray0 = pyrRay.GetConstImage(0);
    tdp::Image<uint32_t> z0 = pyrZ.GetImage(0);
    tdp::Image<uint8_t> mask0 = pyrMask.GetImage(0);
    GetAssocOcclusion(pl_w, covs, pc0, ray0, T_cw, numSigmaOclusion,
        z0, mask0, *ids[0]);
    for (size_t lvl=1; lvl < LEVELS; ++lvl) {
      tdp::Image<tdp::Vector3fda> pc0 = pyrPc.GetConstImage(lvl-1);
      tdp::Image<uint32_t> z0 = pyrZ.GetImage(lvl-1);
      tdp::Image<uint8_t> mask0 = pyrMask.GetImage(lvl-1);
      tdp::Image<tdp::Vector3fda> pc1 = pyrPc.GetConstImage(lvl);
      tdp::Image<uint32_t> z1 = pyrZ.GetImage(lvl);
      tdp::Image<uint8_t> mask1 = pyrMask.GetImage(lvl);
      tdp::Vector4ida zs;
      tdp::Vector4fda ds;
      for (size_t v=0; v<pc1.h_; ++v)  {
        for (size_t u=0; u<pc1.w_; ++u)  {
          zs(0) = z0(2*u,2*v);
          zs(1) = z0(2*u,2*v+1);
          zs(2) = z0(2*u+1,2*v);
          zs(3) = z0(2*u+1,2*v+1);
          ds(0) = pc0(2*u,2*v)(2);
          ds(1) = pc0(2*u,2*v+1)(2);
          ds(2) = pc0(2*u+1,2*v)(2);
          ds(3) = pc0(2*u+1,2*v+1)(2);
          ds(0) = ds(0) < dMin || zs(0) == 0 ? 9999. : ds(0);
          ds(1) = ds(1) < dMin || zs(1) == 0 ? 9999. : ds(1);
          ds(2) = ds(2) < dMin || zs(2) == 0 ? 9999. : ds(2);
          ds(3) = ds(3) < dMin || zs(3) == 0 ? 9999. : ds(3);
          int32_t id = 0;
          float minD = ds.minCoeff(&id);
          z1(u,v) = minD > dMax ? 0 : zs(id);
          if (z1(u,v) > 0) {
            mask1(u,v) = 255;
            ids[lvl]->push_back(z1(u,v)-1);
          } else {
            mask1(u,v) = 0;
          }
        }
      }
    }
  }

  void Associate(const Image<Vector3fda>& pc_w,
      SE3f T_cw, float dMin, float dMax, uint32_t numElems) {
    if (vbo_.num_elements != pc_w.Area()) {
      vbo_.Reinitialise(pangolin::GlArrayBuffer,pc_w.Area(),GL_FLOAT,3,
        GL_DYNAMIC_DRAW);
    }
    vbo_.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
    return Associate(vbo_, T_cw, dMin, dMax, numElems);
  };

  void Associate(pangolin::GlBuffer& vbo,
      SE3f T_cw, float dMin, float dMax) {
    return Associate(vbo, T_cw, dMin, dMax, vbo.num_elements);
  }

  void Associate(const Image<Vector3fda>& pc_w,
      SE3f T_cw, float dMin, float dMax) {
    return Associate(pc_w, T_cw, dMin, dMax, pc_w.Area());
  }
 
  size_t w_, h_;
  CameraBase<float,D,Derived> cam_;
  pangolin::GlTexture tex_;
 private:
  pangolin::GlBuffer vbo_;
  pangolin::GlRenderBuffer render_;
  pangolin::GlFramebuffer fbo_;

};



}
