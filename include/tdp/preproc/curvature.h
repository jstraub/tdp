
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
//      for (size_t k=0; k<ids.size(); k++) std::cout << ids[k] << " "; std::cout << std::endl;
      std::sort(ids.begin(), ids.end());
//      for (size_t k=0; k<ids.size(); k++) std::cout << ids[k] << " "; std::cout << std::endl;
      bool checkAllDuplicate = true;
      for (size_t k=0; k<ids.size(); k+=2) {
        if (ids[k] != ids[k+1]) {
          checkAllDuplicate = false; 
          break;
        }
      }
      if (!checkAllDuplicate) {
//        std::cout << "found non-duplicate ids; aborting" << i << std::endl;
        neigh.erase(i);
        continue;
      }
      auto end = std::unique(ids.begin(), ids.end());
//      for (size_t k=0; k<std::distance(ids.begin(),end); k++) std::cout << ids[k] << " "; std::cout << std::endl;

      // setup  orthogonal cosy
      Vector3fda dirx;
      RejectAfromB(vert[ids[0]] - x0, n[i], dirx);
      dirx.normalize();
      Vector3fda diry = dirx.cross(n[i]).normalized();

      auto angleToDir0 = [&](uint32_t a) -> float {
          Vector3fda dira;
          RejectAfromB(vert[a] - x0, n[i], dira);
          return atan2(LengthOfAonB(dira,diry), LengthOfAonB(dira,dirx));
      };

      std::sort(ids.begin(), end,[&](uint32_t a, uint32_t b){
            return angleToDir0(a) < angleToDir0(b);
          });
//      for (size_t k=0; k<std::distance(ids.begin(),end); k++) std::cout << ids[k] << " "; std::cout << std::endl;
//      for (size_t k=0; k<std::distance(ids.begin(),end); k++) 
//        std::cout << angleToDir0(ids[k])*180./M_PI << " "; 
//      std::cout << std::endl;
      neigh[i] = std::vector<uint32_t>(ids.begin(), end);
    }
    Progress(i, vert.w_);
  }
}

void ComputeCurvature(
    const Image<Vector3fda>& vert, 
    const Image<Vector3uda>& tri, 
    const std::map<uint32_t,std::vector<uint32_t>>& neigh,
    Image<Vector3fda>& meanCurv,
    Image<float>& gausCurv,
    Image<float>& area
    ) {
  meanCurv.Fill(Vector3fda(NAN,NAN,NAN));
  gausCurv.Fill(NAN);
  area.Fill(NAN);
  size_t c=0;
  for (const auto& it : neigh) {
    const uint32_t i = it.first;
    const std::vector<uint32_t>& ids = it.second;  
    const Vector3fda& xi = vert[i];
    Vector3fda mc(0,0,0);
    float gc = 0.;
    float A = 0;
    std::cout << "@ vert " << i << std::endl;
    for (int32_t k=0; k<(int32_t)ids.size(); ++k) {
      int32_t l = (k-1+ids.size())%ids.size();
      int32_t r = (k+1+ids.size())%ids.size();
      const Vector3fda& xj = vert[ids[k]];
      const Vector3fda& xl = vert[ids[l]];
      const Vector3fda& xr = vert[ids[r]];
      float dotAlpha = DotABC(xi,xl,xj);
      float dotBeta = DotABC(xi,xr,xj);
      float alpha = acos(dotAlpha);
      float beta = acos(dotBeta);

      std::cout << ids[k]-i << " " << ids[l]-i << " " << ids[r]-i << std::endl;

      float gamma = acos(DotABC(xl,xi,xj));
      if (gamma < 0.5*M_PI && alpha < 0.5*M_PI && alpha+gamma > 0.5*M_PI) {
        // non-obtuse triangle -> voronoi formula
        A += 0.125*(dotAlpha/sin(alpha)+dotBeta/sin(beta)) *(xi-xj).norm();
        std::cout << "non-obtuse: " 
          << gamma * 180./M_PI << ": " 
          << alpha * 180./M_PI << ": " 
          << (dotAlpha/sin(alpha)+dotBeta/sin(beta)) << " " 
          << (xi-xj).norm() << " " 
          << 0.125*(dotAlpha/sin(alpha)+dotBeta/sin(beta))*(xi-xj).norm() << std::endl;
      } else if (gamma > 0.5*M_PI) {
        A += 0.25*((xl-xi).cross(xj-xi)).norm();
        std::cout << "obtuse at xi: " 
          << gamma * 180./M_PI << ": " 
          << alpha * 180./M_PI << ": " 
          << (xl-xi).norm() << " " 
          << (xj-xi).norm() << " " 
          << 0.25*((xl-xi).cross(xj-xi)).norm() << std::endl;
      } else {
        A += 0.125*((xl-xi).cross(xj-xi)).norm();
        std::cout << "obtuse at other than xi: " 
          << gamma * 180./M_PI << ": " 
          << alpha * 180./M_PI << ": " 
          << (xl-xi).norm() << " " 
          << (xj-xi).norm() << " " 
          << 0.125*((xl-xi).cross(xj-xi)).norm() << std::endl;
      }
      mc += (dotAlpha/sin(alpha)+dotBeta/sin(beta))*(xi-xj);
      gc += gamma;
    }
    if (gc!=gc) 
      std::cerr << "gauss curvature nan at " << c << std::endl;
    if (1e-6 < A) {
      meanCurv[i] = mc/(2.*A);
      gausCurv[i] = (2*M_PI-gc)/A;
    }
    area[i] = A;
    Progress(c++, neigh.size());
  }

//  for (size_t i=0; i < vert.w_; ++i) {
//    std::cout << "@vert " << i << ": " << meanCurv[i].transpose()
//      << ", " << gausCurv[i] << std::endl;
//  }
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
    Progress(i, meanCurv.w_);
  }
}

}
