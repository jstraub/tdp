
#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <tdp/eigen/dense.h>
#include <tdp/geometry/vectors.h>
#include <tdp/data/image.h>
#include <tdp/utils/status.h>

namespace tdp {

void ComputeInvertedIndex(const Image<Vector3fda>& vert, 
    const Image<Vector3uda>& tri, 
    std::map<uint32_t,std::vector<uint32_t>>& invertedTri
    ) {
  invertedTri.clear();
  for (size_t i=0; i < vert.w_; ++i) {
    std::vector<uint32_t> ids;
    for (size_t j=0; j < tri.w_; ++j) {
      if ((tri[j].array() == i).any()) {
        ids.push_back(j); 
      }
    }
    if (ids.size() > 0) {
      invertedTri[i] = ids;
    }
  }
}

void ComputeNeighborhood(
    const Image<Vector3fda>& vert, 
    const Image<Vector3uda>& tri, 
    Image<Vector3fda>& n, 
    std::map<uint32_t,std::vector<uint32_t>>& neigh
    ) {

  n.Fill(Vector3fda::Zero());
  neigh.clear();
  for (size_t j=0; j < tri.w_; ++j) {
    uint32_t i0 = tri[j](0);
    uint32_t i1 = tri[j](1);
    uint32_t i2 = tri[j](2);
    
    if (neigh.find(i0) != neigh.end()) {
      neigh[i0].push_back(i1);
      neigh[i0].push_back(i2);
    } else {
      neigh[i0] = {i1,i2}; 
    }
    if (neigh.find(i1) != neigh.end()) {
      neigh[i1].push_back(i0);
      neigh[i1].push_back(i2);
    } else {
      neigh[i1] = {i0,i2}; 
    }
    if (neigh.find(i2) != neigh.end()) {
      neigh[i2].push_back(i0);
      neigh[i2].push_back(i1);
    } else {
      neigh[i2] = {i0,i1}; 
    }

    n[i0] += (vert[i1]-vert[i0]).cross(vert[i2]-vert[i0]).normalized();
    n[i1] += (vert[i0]-vert[i1]).cross(vert[i2]-vert[i1]).normalized();
    n[i2] += (vert[i0]-vert[i2]).cross(vert[i1]-vert[i2]).normalized();

    Progress(j, tri.w_);
  }

  for (size_t i=0; i < vert.w_; ++i) {
    std::vector<uint32_t>& ids = neigh[i];
    if (ids.size() > 0) {
//      std::cout << "# ids: " << ids.size() << std::endl;
      const Vector3fda& x0 = vert[i];
      n[i] = Vector3fda::Zero();
      for (size_t k=0; k<ids.size(); k+=2) {
        Vector3fda ni = (vert[ids[k]]-x0).cross(vert[ids[k+1]]-x0);
        n[i] += ni/ni.norm();
      }
      std::sort(ids.begin(), ids.end());
      auto end = std::unique(ids.begin(), ids.end());

      // sort according to angle from dir0 in tangent plane defined by n
      Vector3fda dir0;
      RejectAfromB(vert[ids[0]] - x0, n[i], dir0);
      std::sort(ids.begin(), end,[&](uint32_t a, uint32_t b){
          Vector3fda dira, dirb, diray, dirby;
          RejectAfromB(vert[a] - x0, n[i], dira);
          RejectAfromB(vert[b] - x0, n[i], dirb);
          RejectAfromB(dira,dir0, diray);
          RejectAfromB(dirb,dir0, dirby);
          float alphaa = atan2(diray.norm(), LengthOfAonB(dira,dir0));
          float alphab = atan2(dirby.norm(), LengthOfAonB(dirb,dir0));
          return alphaa < alphab;
          });
      neigh[i] = std::vector<uint32_t>(ids.begin(), end);
    }
    if (i%1000 == 0)
      std::cout << i << " of " << vert.w_ << std::endl;
  }
}

void ComputeCurvature(
    const Image<Vector3fda>& vert, 
    const Image<Vector3uda>& tri, 
    const std::map<uint32_t,std::vector<uint32_t>>& neigh,
    Image<Vector3fda>& meanCurv,
    Image<float>& gausCurv
    ) {
  meanCurv.Fill(Vector3fda(NAN,NAN,NAN));
#pragma omp parallel for
  for (const auto& it : neigh) {
    const uint32_t i = it.first;
    const std::vector<uint32_t>& ids = it.second;  
    const Vector3fda& xi = vert[i];
    Vector3fda mc(0,0,0);
    float gc = 0.;
    float A = 0;
    for (size_t k=0; k<ids.size(); ++k) {
      const Vector3fda& xj = vert[ids[k]];
      const Vector3fda& xl = vert[(ids[k]-1+ids.size())%ids.size()];
      const Vector3fda& xr = vert[(ids[k]+1+ids.size())%ids.size()];
      float dotAlpha = DotABC(xi,xl,xj);
      float dotBeta = DotABC(xi,xr,xj);
      float alpha = acos(dotAlpha);
      float beta = acos(dotBeta);

      float gamma = acos(DotABC(xl,xi,xj));
      if (gamma < 0.5*M_PI || alpha < 0.5*M_PI || alpha+gamma > 0.5*M_PI) {
        // non-obtuse triangle -> voronoi formula
        A += 0.125*(dotAlpha/sin(alpha)+dotBeta/sin(beta)) *(xi-xj).norm();
      } else if (gamma > 0.5*M_PI) {
        A += 0.25*((xl-xi).cross(xj-xi)).norm();
      } else {
        A += 0.125*((xl-xi).cross(xj-xi)).norm();
      }
      mc += (dotAlpha/sin(alpha)+dotBeta/sin(beta))*(xi-xj);
      gc += gamma;
    }
    meanCurv[i] = mc/(2.*A);
    gausCurv[i] = (2*M_PI-gc)/A;
  }
}

void ComputePrincipalCurvature(
    const Image<Vector3fda>& meanCurv,
    const Image<float>& gausCurv,
    Image<Vector2fda>& principalCurv
    ) {
  for (size_t i=0; i<meanCurv.w_; ++i) {
    float kappaH = 0.5*meanCurv[i].norm();
    float sqrtDelta = sqrt(std::max(0.f,kappaH*kappaH-gausCurv[i]));
    principalCurv[i](0) = kappaH + sqrtDelta;
    principalCurv[i](1) = kappaH - sqrtDelta;
  }
}

}
