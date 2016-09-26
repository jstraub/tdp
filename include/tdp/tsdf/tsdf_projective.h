#pragma once

#include <tdp/data/volume.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera.h>
#include <tdp/data/image.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

// Projective TSDF puts the TSDF in a depth map of a reference camera;
// the depth values are spaced according to linear inverse depth
void AddToProjectiveTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu);

void RayTraceProjectiveTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu);

}
