#pragma once

#include <tdp/volume.h>
#include <tdp/camera.h>
#include <tdp/image.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

// The TSDF is constructed in a cube with voxel side-length dGrid and
// origin at grid0 in reference coordinates.
void AddToTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);

// get depth image and surface normals from pose T_rd
void RayTraceTSDF(Volume<float> tsdf, Image<float> d, 
    Image<Vector3fda> n, 
    SE3<float> T_rd, Camera<float>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);
// get point cloud and surface normals from T_rd in reference
// coordiante frame
void RayTraceTSDF(Volume<float> tsdf, 
    Image<Vector3fda> pc_r, 
    Image<Vector3fda> n_r, 
    SE3<float> T_rd, Camera<float>camD,
    Vector3fda grid0, Vector3fda dGrid,
    float mu);

// Projective TSDF puts the TSDF in a depth map of a reference camera;
// the depth values are spaced according to linear inverse depth
void AddToProjectiveTSDF(Volume<float> tsdf, Volume<float> W, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu);

void RayTraceProjectiveTSDF(Volume<float> tsdf, Image<float> d, 
    SE3<float> T_rd, Camera<float> camR, Camera<float>camD,
    float rho0, float drho, float mu);

}
