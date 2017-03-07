#pragma once 
#include <tdp/data/volume.h>
#include <tdp/tsdf/tsdf.h>
#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/marching_cubes/CIsoSurface.h>
#include <pangolin/gl/glvbo.h>

namespace tdp {

bool ComputeMesh(
  const Volume<TSDFval>& tsdf,
  const Vector3fda& grid0,
  const Vector3fda& dGrid,
  SE3f& T_wg, // transformation from grid coordinate system to world
  pangolin::GlBuffer& vbo,
  pangolin::GlBuffer& cbo,
  pangolin::GlBuffer& ibo,
  float wThr,
  float fThr
    ) {
  CIsoSurface surface;
  surface.GenerateSurface(&tsdf, 0.0f, dGrid(0), dGrid(1), dGrid(2), wThr, fThr);
  if (!surface.IsSurfaceValid()) {
    std::cerr << "Unable to generate surface" << std::endl;
    return false;
  }
  size_t nVertices = surface.numVertices();
  size_t nTriangles = surface.numTriangles();
  std::cout << "Number of Vertices: " << nVertices << std::endl;
  std::cout << "Number of Triangles: " << nTriangles << std::endl;

  ManagedHostImage<Vector3fda> vertexStore(nVertices,1);
  ManagedHostImage<Vector3bda> colorStore(nVertices,1);
  ManagedHostImage<Vector3uda> indexStore(nTriangles,1);

  surface.getVertices((float*)vertexStore.ptr_);
  surface.getIndices((uint32_t*)indexStore.ptr_);
  surface.getColors((uint8_t*)colorStore.ptr_);

  std::cout << "Mesh Vol: " << surface.getVolume() << std::endl;

  for (size_t i=0; i<vertexStore.Area(); ++i)
    vertexStore[i] = T_wg*(vertexStore[i] + grid0);

  vbo.Reinitialise(pangolin::GlArrayBuffer, nVertices,  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  cbo.Reinitialise(pangolin::GlArrayBuffer, nVertices,
      GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
  ibo.Reinitialise(pangolin::GlElementArrayBuffer, nTriangles,
      GL_UNSIGNED_INT,  3, GL_DYNAMIC_DRAW);
  vbo.Upload((float*)vertexStore.ptr_, vertexStore.SizeBytes(), 0);
  cbo.Upload((uint8_t*)colorStore.ptr_,  colorStore.SizeBytes(), 0);
  ibo.Upload((uint32_t*)indexStore.ptr_,  indexStore.SizeBytes(), 0);

  return true;
}

}
