#pragma once
#include <tdp/data/image.h>
#include <tdp/camera/camera.h>
#include <tdp/data/pyramid.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

void Depth2PCGpu(
    const Image<float>& d,
    const Camera<float>& cam,
    Image<Vector3fda>& pc_c
    );

void Depth2PCGpu(
    const Image<float>& d,
    const Camera<float>& cam,
    const SE3<float>& T_rc,
    Image<Vector3fda>& pc_r
    );

void Depth2PC(
    const Image<float>& d,
    const Camera<float>& cam,
    Image<Vector3fda>& pc
    );

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

template<int LEVELS>
void Depth2PCsGpu(
    Pyramid<float,LEVELS>& d,
    const Camera<float>& cam,
    Pyramid<Vector3fda,LEVELS>& pc
    ) {
  Camera<float> camLvl = cam;
  for (size_t lvl=0; lvl<LEVELS; ++lvl) {
    Image<float> d_i = d.GetImage(lvl);
    Image<Vector3fda> pc_i = pc.GetImage(lvl);

//    std::cout << d_i.Description() << " " 
//      << camLvl.params_.transpose() << std::endl;
    Depth2PCGpu(d_i, camLvl, pc_i);
    camLvl = ScaleCamera<float>(camLvl,0.5);
  }
}

}
