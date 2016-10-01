#pragma once
#include <tdp/data/image.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera.h>
#include <tdp/data/pyramid.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

/// Transform a PC by a rigid transformation T_rc in place. (on GPU)
void TransformPc(
    const SE3f& T_rc,
    Image<Vector3fda>& pc_c
    );
/// Rotate a PC by R_rc in place. (on GPU)
void TransformPc(
    const SO3f& R_rc,
    Image<Vector3fda>& pc_c
    );

void L2Distance(
    Image<Vector3fda>& pcA,
    Image<Vector3fda>& pcB,
    Image<float>& dist
    );
/// Convert from depth image to point cloud in camera coords.
template<int D, typename Derived>
void Depth2PCGpu(
    const Image<float>& d,
    const CameraBase<float,D,Derived>& cam,
    Image<Vector3fda>& pc_c
    );
/// Convert from depth image to point cloud and transform into
/// reference Cosy via T_rc.
template<int D, typename Derived>
void Depth2PCGpu(
    const Image<float>& d,
    const CameraBase<float,D,Derived>& cam,
    const SE3<float>& T_rc,
    Image<Vector3fda>& pc_r
    );
/// Convert from depth image to point cloud in camera coords.
template<int D, typename Derived>
void Depth2PC(
    const Image<float>& d,
    const CameraBase<float,D,Derived>& cam,
    Image<Vector3fda>& pc
    ) {
  for (size_t v=0; v<pc.h_; ++v)
    for (size_t u=0; u<pc.w_; ++u) 
      if (u<d.w_ && v<d.h_) {
        pc(u,v) = cam.Unproject(u,v,d(u,v));
      }
}

template<int LEVELS>
void TransformPc(
    const SE3f& T_rc,
    Pyramid<Vector3fda,LEVELS>& pc
    ) {
  for (size_t lvl=0; lvl<LEVELS; ++lvl) {
    Image<Vector3fda> pc_i = pc.GetImage(lvl);
    TransformPc(T_rc, pc_i);
  }
}

template<int LEVELS>
void TransformPc(
    const SO3f& R_rc,
    Pyramid<Vector3fda,LEVELS>& pc
    ) {
  for (size_t lvl=0; lvl<LEVELS; ++lvl) {
    Image<Vector3fda> pc_i = pc.GetImage(lvl);
    TransformPc(R_rc, pc_i);
  }
}

template<int LEVELS>
void Depth2PCs(
    Pyramid<float,LEVELS>& d,
    const Camera<float>& cam,
    Pyramid<Vector3fda,LEVELS>& pc
    ) {
  for (size_t lvl=0; lvl<LEVELS; ++lvl) {
    Image<float> d_i = d.GetImage(lvl);
    Image<Vector3fda> pc_i = pc.GetImage(lvl);
    Depth2PC(d_i, cam, pc_i);
  }
}

template<int LEVELS, int D, typename Derived>
void Depth2PCsGpu(
    Pyramid<float,LEVELS>& d,
    const CameraBase<float,D,Derived>& cam,
    Pyramid<Vector3fda,LEVELS>& pc
    ) {
  CameraBase<float,D,Derived> camLvl = cam;
  for (size_t lvl=0; lvl<LEVELS; ++lvl) {
    Image<float> d_i = d.GetImage(lvl);
    Image<Vector3fda> pc_i = pc.GetImage(lvl);

//    std::cout << d_i.Description() << " " 
//      << camLvl.params_.transpose() << std::endl;
    Depth2PCGpu<D,Derived>(d_i, camLvl, pc_i);
    camLvl = ScaleCamera<float>(camLvl,0.5);
  }
}

}
