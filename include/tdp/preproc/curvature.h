
#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include <tdp/eigen/dense.h>
#include <tdp/geometry/vectors.h>
#include <tdp/data/image.h>
#include <tdp/utils/status.h>

namespace tdp {

/// Compute Mean curvature via graph laplacian (implicitly)
bool MeanCurvature(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& c
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
//    c = pc(u0,v0);
//    Eigen::VectorXf norms = Eigen::VectorXf::Zero(4*W*W);
//    size_t i=0;
//    for (size_t u=u0-W; u<u0+W; ++u) {
//      for (size_t v=v0-W; v<v0+W; ++v) {
//        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
//          norms(i) = (c-pc(u,v)).squaredNorm();
//        }
//        i++;
//      }
//    }
//    if (i<4) return false;
//    float scale = norms.maxCoeff()/9.;
//
//    float sum = 0.;
//    i = 0;
//    for (size_t u=u0-W; u<u0+W; ++u) {
//      for (size_t v=v0-W; v<v0+W; ++v) {
//        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
//          sum += exp(-norms(i)/scale);
//        }
//        i++;
//      }
//    }
////    float sum = (-norms.array()/scale).exp().sum();
//    i=0;
//    float sum2 = 0.;
//    for (size_t u=u0-W; u<u0+W; ++u) {
//      for (size_t v=v0-W; v<v0+W; ++v) {
//        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
//          c -= exp(-norms(i)/scale)/sum * pc(u,v);
//          sum2 += exp(-norms(i)/scale)/sum;
////          std::cout << " " << exp(-norms(i)/scale)/sum;
//        }
//        i++;
//      }
//    }
//    std::cout << W << " " << i << " " << scale 
//      << " " << sum 
//      << " " << sum2
//      << " " << c.norm()
//      << std::endl;
//    if (c.norm() < 0.1) {
      c = pc(u0,v0);
      Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
      for (size_t u=u0-W; u<u0+W; ++u) {
        for (size_t v=v0-W; v<v0+W; ++v) {
          if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
            S += (c-pc(u,v))*(c-pc(u,v)).transpose();
          }
        }
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(S);

      int id = 0;
      float eval = eig.eigenvalues().minCoeff(&id);
      c = eig.eigenvectors().col(id);
//      std::cout << eig.eigenvalues().transpose() << std::endl;
//      std::cout << eval << std::endl;
//    }
//    std::cout << std::endl;
    return true;
  }
  return false;
}


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

    // sum up surface normals all pointing into the same direction
    Vector3fda ni0 = ((vert[i1]-vert[i0]).cross(vert[i2]-vert[i0])).normalized();
    Vector3fda ni1 = ((vert[i0]-vert[i1]).cross(vert[i2]-vert[i1])).normalized();
    Vector3fda ni2 = ((vert[i0]-vert[i2]).cross(vert[i1]-vert[i2])).normalized();
    if (n[i0].norm() > 0. && n[i0].normalized().dot(ni0) < 0) {
      n[i0] -= ni0;
    } else {
      n[i0] += ni0;
    }
    if (n[i1].norm() > 0. && n[i1].normalized().dot(ni1) < 0) {
      n[i1] -= ni1;
    } else {
      n[i1] += ni1;
    }
    if (n[i2].norm() > 0. && n[i2].normalized().dot(ni2) < 0) {
      n[i2] -= ni2;
    } else {
      n[i2] += ni2;
    }

    Progress(j, tri.w_);
  }

  for (size_t i=0; i < vert.w_; ++i) {
    n[i].normalize();
    std::vector<uint32_t>& ids = neigh[i];
    if (ids.size() > 0) {
//      std::cout << "# ids: " << ids.size() << std::endl;
      const Vector3fda& x0 = vert[i];
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
//    std::cout << "@ vert " << i << std::endl;
    for (int32_t k=0; k<(int32_t)ids.size(); ++k) {
      const int32_t l = (k-1+ids.size())%ids.size();
      const int32_t r = (k+1+ids.size())%ids.size();
      const Vector3fda& xj = vert[ids[k]];
      const Vector3fda& xl = vert[ids[l]];
      const Vector3fda& xr = vert[ids[r]];
      const float dotAlpha = DotABC(xi,xl,xj);
      const float dotBeta = DotABC(xi,xr,xj);
      const float alpha = acos(dotAlpha);
      const float gamma = acos(DotABC(xl,xi,xj));

//      std::cout << int32_t(ids[k])-int32_t(i) 
//        << " " << int32_t(ids[l])-int32_t(i)
//        << " " << int32_t(ids[r])-int32_t(i) << std::endl;

//        float b = (dotAlpha/sin(alpha)+dotBeta/sin(beta));
      float b = std::max(0.f,dotAlpha/sqrtf(1.f-dotAlpha*dotAlpha)+dotBeta/sqrtf(1.f-dotBeta*dotBeta));
//      std::cout << std::setprecision(6);
      if (gamma < 0.5*M_PI && alpha < 0.5*M_PI && alpha+gamma > 0.5*M_PI) {
        // non-obtuse triangle -> voronoi formula
        A += 0.125* b *(xi-xj).norm();
//        std::cout << "non-obtuse: " 
//          << gamma * 180./M_PI << ": " 
//          << alpha * 180./M_PI << ": " 
//          << b << " " 
//          << (xi-xj).norm() << " " 
//          << " dA=" << 0.125*b*(xi-xj).norm() 
//          << "xis: " << xi.transpose() << ", " << xj.transpose()
//          << std::endl;
      } else if (gamma > 0.5*M_PI) {
        A += 0.25*((xl-xi).cross(xj-xi)).norm();
//        std::cout << "obtuse at xi: " 
//          << gamma * 180./M_PI << ": " 
//          << alpha * 180./M_PI << ": " 
//          << (xl-xi).norm() << " " 
//          << (xj-xi).norm() << " " 
//          << " dA=" << 0.25*((xl-xi).cross(xj-xi)).norm() 
//          << " dxis: " << (xl- xi).transpose() 
//          << ", " <<  (xj-xi).transpose() 
//          << ", " << (xl-xi).cross(xj-xi).transpose()
//          << std::endl;
      } else {
        A += 0.125*((xl-xi).cross(xj-xi)).norm();
//        std::cout << "obtuse at other than xi: " 
//          << gamma * 180./M_PI << ": " 
//          << alpha * 180./M_PI << ": " 
//          << (xl-xi).norm() << " " 
//          << (xj-xi).norm() << " " 
//          << 0.125*((xl-xi).cross(xj-xi)).norm() << std::endl;
      }
      mc += b*(xi-xj);
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
