#pragma once

#include <tdp/volume.h>
#include <tdp/camera.h>
#include <tdp/image.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu);

void RayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu);

}
