
#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <tdp/eigen/dense.h>
#include <tdp/data/image.h>

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
  neigh.clear();
  for (size_t i=0; i < vert.w_; ++i) {
    std::vector<uint32_t> ids;
    n[i] = Vector3fda(NAN,NAN,NAN);
    for (size_t j=0; j < tri.w_; ++j) {
      if (tri[j](0) == i) {
        ids.push_back(tri[j](1)); 
        ids.push_back(tri[j](2)); 
      } else if (tri[j](1) == i) {
        ids.push_back(tri[j](0)); 
        ids.push_back(tri[j](2)); 
      } else if (tri[j](2) == i) {
        ids.push_back(tri[j](0)); 
        ids.push_back(tri[j](1)); 
      }
    }
    if (ids.size() > 0) {
      Vector3fda& x0 = vert[i];
      n[i] = Vector3fda::Zero();
      for (size_t k=0; k<ids.size(); k+=2) {
        Vector3fda ni = (vert[ids[k]]-x0).cross(vert[ids[k+1]]-x0);
        n[i] += ni/ni.norm();
      }
      std::sort(ids.begin(), ids.end());
      auto end = std::unique(ids.begin(), ids.end());

      // sort according to angle from dir0 in tangent plane defined by n
      Vector3fda dir0 = RejectAfromB(vert[ids[0]] - x0, n[i]);
        std::sort(ids.begin(), end,[&](uint32_t a, uint32_t b){
            Vector3fda dira = RejectAfromB(vert[a] - x0, n[i]);
            Vector3fda dirb = RejectAfromB(vert[b] - x0, n[i]);
            float alphaa = atan2(LengthOfAorthoToB(dira,dir0),LengthOfAonB(dira,dir0));
            float alphab = atan2(LengthOfAorthoToB(dirb,dir0),LengthOfAonB(dirb,dir0));
            return alphaa < alphab;
            });
      neigh[i] = std::vector<uint32_t>(ids.begin(), end);
    }
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
  for (const auto& it : neigh) {
    const uint32_t i = it.first;
    const std::vector<uint32_t>& ids = it.second;  
    const Vector3fda& xi = vert[i];
    Vector3fda mc(0,0,0);
    float gc = 0.;
    float A = 0;
#pragma omp parallel for
    for (size_t k=0; k<ids.size(); ++k) {
      Vector3fda& xj = vert[ids[k]];
      Vector3fda& xl = vert[(ids[k]-1+ids.size())%ids.size()];
      Vector3fda& xr = vert[(ids[k]+1+ids.size())%ids.size()];
      float dotAlpha = DotABC(xi,xl,xj);
      float dotBeta = DotABC(xi,xr,xj);
      float alpha = acos(dotAlpha);
      float beta = acos(dotBeta);

      float gamma = acos(DotABC(xl,xi,xj));
      if (gamma < 0.5*M_PI || alpha < 0.5*M_PI || alpha+gamma > 0.5*M_PI) {
        // non-obtuse triangle -> voronoi formula
        A += 0.125*(dotAlpha/sin(alpha)+dotBeta/sin(beta)) *(xi-xj).norm()
      } else if (gamma > 0.5*M_PI) {
        A += 0.25*((xl-xi).cross(xj-xi)).norm();
      } else {
        A += 0.125*((xl-xi).cross(xj-xi)).norm();
      }
      mc += (dotAlpha/sin(alpha)+dotBeta/sin(beta))*(xi-xj);
      gc += gamma;
    }
    meanCurv[i] = mc/(2.*A[i]);
    gausCurv[i] = (2*M_PI-gc)/A[i];
  }
}

}
