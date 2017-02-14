/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <thread>
#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/gldraw.h>

#include <Eigen/Dense>
#include <tdp/camera/rig.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/data/managed_volume.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/volume.h>
#include <tdp/data/circular_buffer.h>
#include <tdp/gl/gl_draw.h>
#include <tdp/gui/gui_base.hpp>
#include <tdp/gui/quickView.h>
#include <tdp/icp/icp.h>
#include <tdp/icp/icpRot.h>
#include <tdp/icp/icpGrad3d.h>
#include <tdp/icp/icpTexture.h>
#include <tdp/manifold/SE3.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/preproc/depth.h>
#include <tdp/preproc/normals.h>
#include <tdp/preproc/pc.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/manifold/SO3.h>
#include <tdp/preproc/grad.h>
#include <tdp/preproc/grey.h>
#include <tdp/preproc/mask.h>
#include <tdp/camera/ray.h>
#include <tdp/preproc/curvature.h>
#include <tdp/geometry/cosy.h>
#include <tdp/geometry/vectors.h>
#include <tdp/gl/shaders.h>
#include <tdp/utils/colorMap.h>
#include <tdp/camera/photometric.h>
#include <tdp/clustering/dpvmfmeans_simple.hpp>
#include <tdp/clustering/managed_dpvmfmeans_simple.hpp>
#include <tdp/features/brief.h>
#include <tdp/features/fast.h>
#include <tdp/preproc/blur.h>
#include <tdp/gl/render.h>
#include <tdp/gl/labels.h>
#include <tdp/preproc/convert.h>
#include <tdp/preproc/plane.h>
#include <tdp/features/lsh.h>
#include <tdp/utils/timer.hpp>
#include <tdp/camera/projective_labels.h>
#include <tdp/ransac/ransac.h>
#include <tdp/utils/file.h>

#include <tdp/sampling/sample.hpp>
#include <tdp/sampling/vmf.hpp>
#include <tdp/sampling/vmfPrior.hpp>
#include <tdp/sampling/normal.hpp>

//#include "planeHelpers.h"
//#include "icpHelper.h"
#include "visHelper.h"

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

#define kNN 12
#define MAP_SIZE 1000000

namespace tdp {

typedef Eigen::Matrix<float,kNN,1,Eigen::DontAlign> VectorkNNfda;
typedef Eigen::Matrix<int32_t,kNN,1,Eigen::DontAlign> VectorkNNida;

bool ExtractClosestBrief(
    const Image<Vector3fda>& pc, 
    const Image<uint8_t>& grey,
    const Image<Vector2ida>& pts,
    const Image<float>& orientation,
    const Vector3fda& pci,
    const Vector3fda& ni,
    const SE3f& T_wc, 
    const CameraT& cam,
    size_t W,
    size_t u, size_t v,
    Brief& feat) {

  feat.pt_(0) = u;
  feat.pt_(1) = v;
  feat.desc_.fill(0);
  // try finding a closeby feature point and get the feature there
  for (size_t j=0; j<pts.Area(); ++j) {
    if ((pts[j].cast<float>() - feat.pt_.cast<float>()).norm() < W) {
      feat.orientation_ = orientation[j];
      if (!tdp::ExtractBrief(grey, feat)) 
        feat.desc_.fill(0);
      else {
        Vector3fda p = pc(pts[j](0), pts[j](1));
        if (!IsValidData(p)) {
          tdp::Rayfda ray(Vector3fda::Zero(), 
              cam.Unproject(pts[j](0), pts[j](1), 1.));
          p = ray.IntersectPlane(pci, ni);
        }
//        std::cout << "FAST feat at " << pts[j].transpose() 
//          << " for " << feat.pt_.transpose() 
//          << " pc " << pc(pts[j](0), pts[j](1)).transpose()
//          << " pIntersect " << p.transpose()
//          << std::endl;
        // TODO: not going to be updated if pl.p_ is !
        feat.p_c_ = T_wc*p; 
      }
      return true;
    }
  }
  return false;
}

void ExtractPlanes(
    const Image<Vector3fda>& pc, 
    const Image<Vector3bda>& rgb,
    const Image<uint8_t>& grey,
    const Image<float>& greyFl,
    const Image<Vector2fda>& gradGrey,
    const Image<uint8_t>& mask, uint32_t W, size_t frame,
    const SE3f& T_wc, 
    const CameraT& cam,
    Image<Vector4fda>& dpc, 
    ManagedHostCircularBuffer<Plane>& pl_w,
    ManagedHostCircularBuffer<Vector3fda>& pc_w,
    ManagedHostCircularBuffer<Vector3fda>& pc0_w,
    ManagedHostCircularBuffer<Vector3bda>& rgb_w,
    ManagedHostCircularBuffer<Vector3fda>& n_w,
    ManagedHostCircularBuffer<float>& rs
    ) {
  Plane pl;
  tdp::Brief feat;
  Vector3fda n, p;
  float curv;
  for (size_t i=0; i<mask.Area(); ++i) {
    if (mask[i] 
        && tdp::IsValidData(pc[i])
        && pc[i].norm() < 5. 
        && 0.3 < pc[i].norm() )  {
//      uint32_t Wscaled = floor(W*pc[i](2));
      uint32_t Wscaled = W;
//      if (tdp::NormalViaScatter(pc, i%mask.w_, i/mask.w_, Wscaled, n)) {
      if (tdp::NormalViaVoting(pc, i%mask.w_, i/mask.w_, Wscaled, 0.29,
            dpc, n, curv, p)) {
//        ExtractClosestBrief(pc, grey, pts, orientation, 
//            p, n, T_wc, cam, Wscaled, i%mask.w_, i/mask.w_, feat);
        pl.p_ = T_wc*p;
        pl.n_ = T_wc.rotation()*n;
        pl.curvature_ = curv;
        pl.rgb_ = rgb[i];
        pl.gradGrey_ = gradGrey[i];
        pl.grey_ = greyFl[i];
        pl.lastFrame_ = frame;
        pl.w_ = 1.;
        pl.numObs_ = 1;
//        pl.feat_ = feat;
//        pl.r_ = 2*W*pc[i](2)/cam.params_(0); // unprojected radius in m
        pl.r_ = p(2)/cam.params_(0); // unprojected radius in m

        pl_w.Insert(pl);
        pc_w.Insert(pl.p_);
        pc0_w.Insert(pl.p_);
        n_w.Insert(pl.n_);
        rgb_w.Insert(pl.rgb_);
        rs.Insert(pl.r_);
      }
    }
  }
}

bool ProjectiveAssoc(const Plane& pl, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Image<Vector3fda>& pc,
    int32_t& u,
    int32_t& v
    ) {
  const tdp::Vector3fda& pc_w = pl.p_;
  Eigen::Vector2f x = cam.Project(T_cw*pc_w);
  u = floor(x(0)+0.5f);
  v = floor(x(1)+0.5f);
  if (0 <= u && u < pc.w_ && 0 <= v && v < pc.h_) {
    if (tdp::IsValidData(pc(u,v))) {
      return true;
    }
  }
  return false;
}


bool EnsureNormal(
    Image<Vector3fda>& pc,
    Image<Vector4fda>& dpc,
    uint32_t W,
    Image<Vector3fda>& n,
    Image<float>& curv,
    int32_t u,
    int32_t v
    ) {
  if (0 <= u && u < pc.w_ && 0 <= v && v < pc.h_) {
    if (tdp::IsValidData(pc(u,v))) {
//      uint32_t Wscaled = floor(W*pc(u,v)(2));
      uint32_t Wscaled = W;
      tdp::Vector3fda ni = n(u,v);
      tdp::Vector3fda pi;
      float curvi;
      if (!tdp::IsValidData(ni)) {
//        if(tdp::NormalViaScatter(pc, u, v, Wscaled, ni)) {
        if(tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, dpc, ni, curvi, pi)) {
          n(u,v) = ni;
          pc(u,v) = pi;
          curv(u,v) = curvi;
          return true;
        }
      } else {
        return true;
      }
    }
  }
  return false;
}

bool ProjectiveAssocNormalExtract(const Plane& pl, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    Image<Vector3fda>& pc,
    uint32_t W,
    Image<Vector4fda>& dpc,
    Image<Vector3fda>& n,
    Image<float>& curv,
    int32_t& u,
    int32_t& v
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  Eigen::Vector2f x = cam.Project(T_cw*pc_w);
  u = floor(x(0)+0.5f);
  v = floor(x(1)+0.5f);
  return EnsureNormal(pc, dpc, W, n, curv, u, v);
}


bool AccumulateP2Pl(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    float distThr, 
    float p2plThr, 
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    float p2pl = n_w.dot(pc_w - pc_c_in_w);
    if (fabs(p2pl) < p2plThr) {
      Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
      Ai.bottomRows<3>() = n_w_in_c; 
      A += Ai * Ai.transpose();
      b += Ai * p2pl;
      err += p2pl;
      return true;
    }
  }
  return false;
}

bool AccumulateP2Pl(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() = n_w_in_c; 
        A += Ai * Ai.transpose();
        b += Ai * p2pl;
        err += p2pl;
        return true;
      }
    }
  }
  return false;
}

/// uses texture as well
bool AccumulateP2Pl(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    float lambda,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float bi=0;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        // p2pl
        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() = n_w_in_c; 
        bi = p2pl;
        A += Ai * Ai.transpose();
        b += Ai * bi;
        err += bi;
        // texture
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_c_in_w);
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci)), 
             Eigen::Matrix3f::Identity();
        Ai = Jse3.transpose() * Jpi.transpose() * pl.gradGrey_;
        bi = grey_ci - pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);
        err += lambda*bi;
        // accumulate
        return true;
      }
    }
  }
  return false;
}

/// uses texture and normal as well
bool AccumulateP2Pl(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    float gamma,
    float lambda,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float bi=0;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        // p2pl
        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() = n_w_in_c; 
        bi = p2pl;
        A += Ai * Ai.transpose();
        b += Ai * bi;
        err += bi;
//        std::cout << "--" << std::endl;
//        std::cout << Ai.transpose() << "; " << bi << std::endl;
        // normal
        Ai.topRows<3>() = -n_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>().fill(0.); 
        bi = n_ci.dot(n_w_in_c) - 1.;
        A += gamma*(Ai * Ai.transpose());
        b += gamma*(Ai * bi);
        err += gamma*bi;
//        std::cout << Ai.transpose() << "; " << bi << std::endl;
        // texture
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_c_in_w);
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci)), 
             Eigen::Matrix3f::Identity();
        Ai = Jse3.transpose() * Jpi.transpose() * pl.gradGrey_;
        bi = grey_ci - pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);
        err += lambda*bi;
//        std::cout << Ai.transpose() << "; " << bi << std::endl;
        // accumulate
        return true;
      }
    }
  }
  return false;
}

// uses texture and projective term
bool AccumulateP2PlProj(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Image<Vector3fda>& pc_c,
    uint32_t u, uint32_t v,
    const Vector3fda& n_ci,
    float grey_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    float lambda,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  const tdp::Vector3fda& pc_ci = pc_c(u,v);
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float bi=0;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        // p2pl projective term
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_c_in_w);
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci)), 
             Eigen::Matrix3f::Identity();
        Eigen::Matrix<float,3,6> Jse3Inv;
        Jse3Inv << SO3mat<float>::invVee(T_wc.rotation().matrix().transpose()*(pc_w-T_wc.translation())), 
             -T_wc.rotation().matrix().transpose();
        
        std::cout << "--" << std::endl;
        // one delta u in image coords translates to delta x = z
//        Eigen::Matrix<float,3,1> tmp(0,0,0);
//        Eigen::Matrix<float,3,1> p_u(pc_ci(2)/cam.params_(0),0,0);
//        Eigen::Matrix<float,3,1> p_v(0,pc_ci(2)/cam.params_(1),0);
////        std::cout << p_u.transpose() << std::endl;
//        RejectAfromB(p_u, n_ci, tmp);
//        p_u = T_wc.rotation()*(tmp * pc_ci(2)/cam.params_(0) / tmp(0));
//        std::cout << tmp.transpose() <<  "; " << tmp.dot(n_ci) << std::endl;
////        std::cout << n_ci.transpose() << std::endl;
//        RejectAfromB(p_v, n_ci, tmp);
////        std::cout << p_u.transpose() << std::endl;
//        p_v = T_wc.rotation()*(tmp * pc_ci(2)/cam.params_(1) / tmp(1));
//        std::cout << tmp.transpose() <<  "; " << tmp.dot(n_ci) << std::endl;
        // could do better by exploiting robust computation if n (above)
        Eigen::Matrix<float,3,1> p_u = T_wc.rotation()*(pc_c(u+1,v) - pc_c(u,v));
        Eigen::Matrix<float,3,1> p_v = T_wc.rotation()*(pc_c(u,v+1) - pc_c(u,v));
        Eigen::Matrix<float,3,2> gradP;
        gradP << p_u, p_v;
        // p2pl projective
        Ai = Jse3Inv.transpose() * Jpi.transpose() * gradP.transpose() * n_w;
        std::cout << Ai.transpose() << std::endl;
//        std::cout << Jse3Inv << std::endl;
//        std::cout << Jpi << std::endl;
//        std::cout << gradP << std::endl;
//        std::cout << n_w.transpose() << std::endl;
//        std::cout << gradP.transpose()*n_w << std::endl;
//        std::cout << p_u.dot(p_v) << std::endl;
//        Ai.fill(0.);
        // p2pl
        Ai.topRows<3>() += pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() += n_w_in_c; 
        std::cout << Ai.transpose() << std::endl;
        bi = p2pl;
        A += Ai * Ai.transpose();
        b += Ai * bi;
        err += bi;
        // texture
        Ai = Jse3.transpose() * Jpi.transpose() * pl.gradGrey_;
        bi = grey_ci - pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);
        err += lambda*bi;
        // accumulate
        return true;
      }
    }
  }
  return false;
}

bool AccumulateRot(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<float,3,3>& N
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float dist = (pc_w - pc_c_in_w).norm();
  if (dist < distThr) {
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        N += n_w * n_ci.transpose();
        return true;
      }
    }
  }
  return false;
}

bool CheckEntropyTermination(const Eigen::Matrix<float,6,6>& A,
    float Hprev,
    float HThr, float condEntropyThr, float negLogEvThr,
    float& H) {

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
  Eigen::Matrix<float,6,1> negLogEv = -eig.eigenvalues().real().array().log();
  H = negLogEv.sum();
  if ((H < HThr || Hprev - H < condEntropyThr) 
      && (negLogEv.array() < negLogEvThr).all()) {
    std::cout <<  " H " << H << " cond H " << (Hprev-H) 
      << " neg log evs: " << negLogEv.transpose() << std::endl;
    return true;
  }
  return false;
}

template<int D>
void AddToSortedIndexList(Eigen::Matrix<int32_t,D,1,Eigen::DontAlign>& ids, 
    Eigen::Matrix<float,D,1,Eigen::DontAlign>&
    values, int32_t id, float value) {
  for(int i=D-1; i>=0; --i) {
    if (value > values[i]) {
//      if (i == D-2) { 
//        values[D-1] = value; 
//        ids[D-1] = id;
//      } else if (i < D-2) {
      if (i < D-1) {
        for (size_t j=D-1; j>i+1; --j) {
          values(j) = values(j-1);
          ids(j) = ids(j-1);
        }
        values(i+1) = value;
        ids(i+1) = id;
//      }
      }
      return;
    }
  }
//  std::cout << value << " " << values.transpose() << std::endl;
  for (size_t j=D-1; j>0; --j) {
    values(j) = values(j-1);
    ids(j) = ids(j-1);
  }
  values(0) = value;
  ids(0) = id;
//        values.bottomRows(D-1) = values.middleRows(i+1,D-1);
//        ids.bottomRows(D-i-2) = ids.middleRows(i+1,D-i-2);
}

//void AddToSortedIndexList(tdp::Vector5ida& ids, tdp::Vector5fda&
//    values, int32_t id, float value) {
//  for(int i=4; i>=0; --i) {
//    if (value > values[i]) {
//      if (i == 3) { 
//        values[4] = value; 
//        ids[4] = id;
//      } else if (i == 2) {
//        values[4] = values[3];
//        values[3] = value; 
//        ids[4] = ids[3];
//        ids[3] = id; 
//      } else if (i == 1) {
//        values[4] = values[3];
//        values[3] = values[2];
//        values[2] = value; 
//        ids[4] = ids[3];
//        ids[3] = ids[2];
//        ids[2] = id; 
//      } else if (i == 0) {
//        values[4] = values[3];
//        values[3] = values[2];
//        values[2] = values[1];
//        values[1] = value; 
//        ids[4] = ids[3];
//        ids[3] = ids[2];
//        ids[2] = ids[1];
//        ids[1] = id; 
//      }
//      return;
//    }
//  }
//  values[4] = values[3];
//  values[3] = values[2];
//  values[2] = values[1];
//  values[1] = values[0];
//  values[0] = value; 
//  ids[4] = ids[3];
//  ids[3] = ids[2];
//  ids[2] = ids[1];
//  ids[1] = ids[0];
//  ids[0] = id; 
//}

}    

int main( int argc, char* argv[] )
{
  std::string input_uri = "openni2://";
  std::string output_uri = "pango://video.pango";
  std::string calibPath = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
//    imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  }

  pangolin::Uri uri = pangolin::ParseUri(input_uri);
  if (!uri.scheme.compare("file")) {
    std::cout << uri.scheme << std::endl; 
    if (pangolin::FileExists(uri.url+std::string("imu.pango"))
     && pangolin::FileExists(uri.url+std::string("video.pango"))) {
//      imu_input_uri = input_uri + std::string("imu.pango");
      tsdfOutputPath = uri.url + tsdfOutputPath;
      input_uri = input_uri + std::string("video.pango");
    } else if (pangolin::FileExists(uri.url+std::string("video.pango"))) {
      input_uri = input_uri + std::string("video.pango");
    } 
  }
  std::vector<std::string> streamTimeStamps;
  if (!uri.scheme.compare("files")) {
    std::cout << uri.scheme << std::endl;
    std::cout << uri.url << std::endl;
    std::string tumDepthFiles = pangolin::PathParent(uri.url,2)+std::string("/depth.txt");
    std::cout << tumDepthFiles << std::endl;
    if (pangolin::FileExists(tumDepthFiles)) {
      std::cout << "found depth files; reading timestamps" << std::endl;
      std::ifstream in(tumDepthFiles);
      while(!in.eof()) {
        std::string time, path;
        if (in.peek() == '#') {
          std::getline(in,path);
          std::cout << "comment " << path << std::endl;
          continue;
        }
        in >> time;
        in >> path;
        streamTimeStamps.push_back(time);
//        std::cout << time << std::endl;
      }
      std::cout << "loaded " << streamTimeStamps.size() 
        << " timestamps" << std::endl;
    }
  }

  std::cout << input_uri << std::endl;
  std::cout << imu_input_uri << std::endl;

  Stopwatch::getInstance().setCustomSignature(82043984912);

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  // optionally connect to IMU if it is found.
//  tdp::ImuInterface* imu = nullptr; 
//  if (imu_input_uri.size() > 0) 
//    imu = tdp::OpenImu(imu_input_uri);
//  if (imu) imu->Start();
//  tdp::ImuInterpolator imuInterp(imu,nullptr);
//  imuInterp.Start();

  tdp::GuiBase gui(1200,800,video);
  gui.container().SetLayout(pangolin::LayoutEqual);

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,true);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
  } else {
    return 2;
  }
  CameraT cam = rig.cams_[rig.rgbStream2cam_[0]];

  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = (w+w%64); // for convolution
  size_t hc = rig.NumCams()*(h+h%64);
  wc += wc%64;
  hc += hc%64;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1280,960,840,840,639.5,479.5,0.1,1000),
//      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  pangolin::OpenGlRenderState normalsCam(
      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.0,-2.2, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);

  pangolin::View& viewAssoc = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewAssoc);

  pangolin::View& viewNormals = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
//  gui.container().AddDisplay(viewNormals);
  viewPc3D.SetLayout(pangolin::LayoutOverlay);
  viewPc3D.AddDisplay(viewNormals);
  viewNormals.SetBounds(0.,0.4,0.6,1.);

  tdp::QuickView viewCurrent(wc, hc);
//  gui.container().AddDisplay(viewCurrent);
  viewPc3D.AddDisplay(viewCurrent);
  viewCurrent.SetBounds(0.,0.3,0.,0.3);

  tdp::QuickView viewGreyGradNorm(wc, hc);
  viewPc3D.AddDisplay(viewGreyGradNorm);
  viewGreyGradNorm.SetBounds(0.,0.3,0.3,0.6);

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewGrey(wc, hc);
  containerTracking.AddDisplay(viewGrey);
  tdp::QuickView viewMask(wc, hc);
  containerTracking.AddDisplay(viewMask);
  gui.container().AddDisplay(containerTracking);


  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
//  pangolin::DataLog logInliers;
//  pangolin::Plotter plotInliers(&logInliers, -100.f,1.f, 0, 130000.f, 
//      10.f, 0.1f);
//  plotters.AddDisplay(plotInliers);
//  pangolin::DataLog logRmse;
//  pangolin::Plotter plotRmse(&logRmse, -100.f,1.f, 0.f,0.2f, 0.1f, 0.1f);
//  plotters.AddDisplay(plotRmse);
  pangolin::DataLog logdH;
  pangolin::Plotter plotdH(&logdH, -100.f,1.f, .5f,1.5f, .1f, 0.1f);
  plotters.AddDisplay(plotdH);
  pangolin::DataLog logObs;
  pangolin::Plotter plotObs(&logObs, -100.f,1.f, 0.f,6.f, .1f, 0.1f);
  plotters.AddDisplay(plotObs);
  pangolin::DataLog logEntropy;
  pangolin::Plotter plotH(&logEntropy, -100.f,1.f, -30.f,0.f, .1f, 0.1f);
  plotters.AddDisplay(plotH);
  pangolin::DataLog logEig;
  pangolin::Plotter plotEig(&logEig, -100.f,1.f, -5.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotEig);
  pangolin::DataLog logEv;
  pangolin::Plotter plotEv(&logEv, -100.f,1.f, -1.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotEv);
  gui.container().AddDisplay(plotters);

  containerTracking.Show(false);
  viewAssoc.Show(false);
  plotters.Show(false);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);
  tdp::ManagedHostImage<float> curv(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);
  tdp::ManagedHostImage<tdp::Vector4fda> dpc(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);

  tdp::ManagedHostImage<uint8_t> grey(w, h);
  tdp::ManagedHostImage<float> greyFl(wc,hc);
  tdp::ManagedDeviceImage<uint8_t> cuGrey(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreyFl(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyFlSmooth(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDu(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDv(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyGradNorm(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyGradTheta(wc,hc);
  tdp::ManagedHostImage<float> greyGradNorm(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector2fda> cuGradGrey(wc,hc);
  tdp::ManagedHostImage<tdp::Vector2fda> gradGrey(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedDeviceImage<uint8_t> cuMask(wc, hc);
  tdp::ManagedHostImage<uint8_t> mask(wc, hc);
  tdp::ManagedHostImage<uint32_t> z(w, h);

  tdp::ManagedHostImage<float> age;


  // ICP stuff
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);
  pangolin::GlBuffer valuebo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,1);

//  tdp::ManagedHostImage<tdp::Vector3fda> pc_c;
//  tdp::ManagedHostImage<tdp::Vector3bda> rgb_c;
//  tdp::ManagedHostImage<tdp::Vector3fda> n_c;

  pangolin::Var<bool> record("ui.record",false,true);
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,10.);

  pangolin::Var<float> subsample("ui.subsample %",0.001,0.0001,.001);
  pangolin::Var<float> scale("ui.scale",0.05,0.1,1);
  pangolin::Var<bool> useFAST("ui.use FAST",false,true);

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> runLoopClosure("ui.run loop closure",false,true);
  pangolin::Var<bool> runLoopClosureGeom("ui.run loop closure geom",false,true);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updatePlanes("ui.update planes",true,true);
  pangolin::Var<bool> updateMap("ui.update map",true,true);

  pangolin::Var<bool> doRegvMF("ui.reg vMF",false,true);
  pangolin::Var<bool> doRegPc0("ui.reg pc0",true,true);
  pangolin::Var<bool> doRegAbsPc("ui.reg abs pc",false,true);
  pangolin::Var<bool> doRegAbsN("ui.reg abs n",true,true);
  pangolin::Var<bool> doRegRelPlZ("ui.reg rel Pl",true,true);
  pangolin::Var<bool> doRegRelNZ("ui.reg rel N",true,true);
  pangolin::Var<bool> doRegRelPlObs("ui.reg rel PlObs",false,true);
  pangolin::Var<bool> doRegRelNObs("ui.reg rel NObs",false,true);
  pangolin::Var<float> lambdaRegDir("ui.lamb Reg Dir",0.01,0.01,1.);
  pangolin::Var<float> lambdaRegPl("ui.lamb Reg Pl",1.0,0.01,10.);
  pangolin::Var<float> lambdaRegPc0("ui.lamb Reg Pc0",0.01,0.01,1.);
  pangolin::Var<float> lambdaMRF("ui.lamb z MRF",.1,0.01,10.);
  pangolin::Var<float> alphaGrad("ui.alpha Grad",.0001,0.0,1.);

  pangolin::Var<bool> pruneAssocByRender("ui.prune assoc by render",true,true);
  pangolin::Var<float> lambdaNs("ui.lamb Ns",0.1,0.0,1.);
  pangolin::Var<float> lambdaTex("ui.lamb Tex",0.0001,0.0,0.1);
  pangolin::Var<bool> useTexture("ui.use Tex in ICP",false,true);
  pangolin::Var<bool> useNormals("ui.use Ns in ICP",true,true);


  pangolin::Var<bool> icpReset("ui.reset icp",true,false);
  pangolin::Var<float> angleUniformityThr("ui.angle unif thr",5, 0, 90);
  pangolin::Var<float> angleThr("ui.angle Thr",15, -1, 90);
//  pangolin::Var<float> angleThr("ui.angle Thr",-1, -1, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.03,0,0.3);
  pangolin::Var<float> distThr("ui.dist Thr",0.1,0,0.3);
  pangolin::Var<float> curvThr("ui.curv Thr",1.,0.01,1.0);
  pangolin::Var<float> assocDistThr("ui.assoc dist Thr",0.1,0,0.3);
  pangolin::Var<float> HThr("ui.H Thr",-12.,-20.,-8.);
  pangolin::Var<float> negLogEvThr("ui.neg log ev Thr",-0.,-2.,1.);
  pangolin::Var<float> condEntropyThr("ui.rel log dH ", 1.e-3,1.e-3,1e-2);
  pangolin::Var<float> icpdRThr("ui.dR Thr",0.25,0.1,1.);
  pangolin::Var<float> icpdtThr("ui.dt Thr",0.01,0.01,0.001);
  pangolin::Var<int> numRotThr("ui.numRot Thr",200, 100, 350);
  pangolin::Var<int> maxIt("ui.max iter",15, 1, 20);

  pangolin::Var<int>   W("ui.W ",9,1,15);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool> showPlanes("ui.show planes",false,true);
  pangolin::Var<bool> showPcModel("ui.show model",false,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",false,true);
  pangolin::Var<bool> showFullPc("ui.show full",true,true);
  pangolin::Var<bool> showNormals("ui.show ns",true,true);
  pangolin::Var<bool> showAge("ui.show age",false,true);
  pangolin::Var<bool> showObs("ui.show # obs",false,true);
  pangolin::Var<bool> showCurv("ui.show curvature",false,true);
  pangolin::Var<bool> showDPvMFlabels("ui.show DPvMF labels",true,true);
  pangolin::Var<bool> showLabels("ui.show labels",true,true);
  pangolin::Var<bool> showSamples("ui.show Samples",false,true);
  pangolin::Var<bool> showSurfels("ui.show surfels",true,true);
  pangolin::Var<bool> showNN("ui.show NN",true,true);
  pangolin::Var<bool> showLoopClose("ui.show loopClose",false,true);
  pangolin::Var<int> step("ui.step",10,0,100);

  pangolin::Var<float> ransacMaxIt("ui.max it",3000,1,1000);
  pangolin::Var<float> ransacThr("ui.thr",0.09,0.01,1.0);
  pangolin::Var<float> ransacInlierThr("ui.inlier thr",6,1,20);

  tdp::SE3f T_wc_0;
  tdp::SE3f T_wc = T_wc_0;
  tdp::SE3f T_wcRansac;
  std::vector<tdp::SE3f> T_wcs;
  Eigen::Matrix<float,6,6> Sigma_mc;
  std::vector<float> logHs;

  gui.verbose = true;
  if (gui.verbose) std::cout << "starting main loop" << std::endl;

  pangolin::GlBuffer vbo_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,3);
  pangolin::GlBuffer nbo_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,3);
  pangolin::GlBuffer rbo(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,1);
  pangolin::GlBuffer lbo(pangolin::GlArrayBuffer,MAP_SIZE,GL_UNSIGNED_SHORT,1);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(MAP_SIZE);
  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  tdp::ManagedHostCircularBuffer<float> rs(MAP_SIZE); // radius of surfels
  rs.Fill(NAN);
  pangolin::GlBuffer cbo_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_UNSIGNED_BYTE,3);
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(MAP_SIZE);
  rgb_w.Fill(tdp::Vector3bda::Zero());

  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNida> nn(MAP_SIZE);
  nn.Fill(tdp::VectorkNNida::Ones()*-1);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> mapObsNum(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> mapObsDot(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> mapObsP2Pl(MAP_SIZE);
  mapObsNum.Fill(tdp::VectorkNNfda::Zero());

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pcSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> numSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc0_w(MAP_SIZE);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> Jn_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> Jp_w(MAP_SIZE);

  tdp::ManagedHostImage<uint16_t> labels;

  std::vector<std::pair<size_t, size_t>> mapNN;
  mapNN.reserve(MAP_SIZE*kNN);

  int32_t iReadCurW = 0;
  size_t frame = 0;

  tdp::ThreadedValue<bool> runSampling(true);
  tdp::ThreadedValue<bool> runTopologyThread(true);
  tdp::ThreadedValue<bool> runMappingThread(true);

  std::mutex pl_wLock;
  std::mutex nnLock;
  std::mutex mapLock;
  std::mutex dpvmfLock;
  std::thread topology([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
    int32_t sizeToRead = 0;
    tdp::VectorkNNfda values;
    while(runTopologyThread.Get()) {
      {
        std::lock_guard<std::mutex> lock(pl_wLock); 
        iRead = pl_w.iRead_;
        iInsert = pl_w.iInsert_;
        sizeToRead = pl_w.SizeToRead();
      }
      if (sizeToRead > 0) {
        values.fill(std::numeric_limits<float>::max());
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        tdp::VectorkNNida& ids = nn[iReadNext];
        tdp::VectorkNNida idsPrev = ids;
        ids = tdp::VectorkNNida::Ones()*(-1);
        for (size_t i=0; i<sizeToRead; ++i) {
          if (i != iReadNext) {
            float dist = (pl.p_-pl_w.GetCircular(i).p_).squaredNorm();
            tdp::AddToSortedIndexList<kNN>(ids, values, i, dist);
//            std::cout << i << ", " << dist << "| " <<  ids.transpose() << " : " << values.transpose() << std::endl;
          }
        }
        // for map constraints
        // TODO: should be updated as pairs are reobserved
        for (int i=0; i<kNN; ++i) {
//            mapObsDot[iReadNext][i] = pl.n_.dot(pl_w[ids[i]].n_);
//            mapObsP2Pl[iReadNext][i] = pl.p2plDist(pl_w[ids[i]].p_);
//            mapObsNum[iReadNext][i] = 1;
          if (ids(i) != idsPrev(i)) {
            mapObsDot[iReadNext][i] = 0.;
            mapObsP2Pl[iReadNext][i] = 0.;
            mapObsNum[iReadNext][i] = 0.;
//            std::cout << "resetting " << iReadNext << " " << i << std::endl;
          }
        }
        // just for visualization
        if (mapNN.size() < kNN*iReadNext) {
          for (int i=0; i<kNN; ++i) 
            mapNN.emplace_back(iReadNext, ids[i]);
        } else {
          for (int i=0; i<kNN; ++i) 
            mapNN[iReadNext*kNN+i] = std::pair<size_t,size_t>(iReadNext, ids[i]);
        }
        iReadNext = (iReadNext+1)%sizeToRead;
        {
          std::lock_guard<std::mutex> lock(nnLock); 
//          mapObsDot.iInsert_ = std::max(iReadNext;
//          mapObsP2Pl.iInsert_ = std::max(iReadNext;
          nn.iInsert_ = std::max(iReadNext, nn.iInsert_);
        }
      }
    };
  });


  uint32_t K = 0;
  std::mutex zsLock;
  std::mutex vmfsLock;
  std::mt19937 rnd(910481);
  float logAlpha = log(.1);
  float tauO = 100.;
//  Eigen::Matrix3f SigmaO = 0.0001*Eigen::Matrix3f::Identity();
//  Eigen::Matrix3f InfoO = 10000.*Eigen::Matrix3f::Identity();
  vMFprior<float> base(Eigen::Vector3f(0,0,1), .01, 0.);
  std::vector<vMF<float,3>> vmfs;
  vmfs.push_back(base.sample(rnd));

  tdp::ManagedHostCircularBuffer<uint16_t> zS(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<uint16_t> zCountS(MAP_SIZE); // count how often the same cluster ID
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nS(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pS(MAP_SIZE);
  nS.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  pS.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  zS.Fill(9999); //std::numeric_limits<uint32_t>::max());
  tdp::ManagedHostCircularBuffer<tdp::Vector4fda> vmfSS(10000);
  vmfSS.Fill(tdp::Vector4fda::Zero());

  std::thread sampling([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
//    std::random_device rd_;
    std::mt19937 rnd(0);
    while(runSampling.Get()) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        iRead = nn.iRead_;
        iInsert = nn.iInsert_;
      }
      pS.iInsert_ = nn.iInsert_;
      nS.iInsert_ = nn.iInsert_;
      // sample normals using dpvmf and observations from planes
      size_t Ksample = vmfs.size();
      vmfSS.Fill(tdp::Vector4fda::Zero());
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        tdp::Vector3fda& ni = nS[i];
        uint16_t& zi = zS[i];
//        tdp::Plane& pl = pl_w[i];
//        Eigen::Vector3f mu = pl.w_*pl.n_*tauO;
//        std::cout << pl.w_ * pl.n_.transpose() << " " 
//          << nSum_w[i].transpose() << std::endl;
        Eigen::Vector3f mu = nSum_w[i]*tauO;
        if (zi < Ksample) {
          mu += vmfs[zi].mu_*vmfs[zi].tau_;
        }
        ni = vMF<float,3>(mu).sample(rnd);
        vmfSS[zi].topRows<3>() += ni;
        vmfSS[zi](3) ++;
      }
      // sample dpvmf labels
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        Eigen::VectorXf logPdfs(Ksample+1);
        Eigen::VectorXf pdfs(Ksample+1);

        tdp::Vector3fda& ni = nS[i];
        uint16_t& zi = zS[i];
        tdp::VectorkNNida& ids = nn[i];

        Eigen::VectorXf neighNs = Eigen::VectorXf::Zero(Ksample);
        for (int i=0; i<kNN; ++i) {
          if (ids[i] > -1  && zS[ids[i]] < Ksample) {
            neighNs[zS[ids[i]]] += 1.f;
          }
        }
        for (size_t k=0; k<Ksample; ++k) {
          logPdfs[k] = lambdaMRF*(neighNs[k]-kNN);
          if (zi == k) {
            logPdfs[k] += log(vmfSS[k](3)-1)+vmfs[k].logPdf(ni);
          } else {
            logPdfs[k] += log(vmfSS[k](3))+vmfs[k].logPdf(ni);
          }
        }
        logPdfs[Ksample] = logAlpha + base.logMarginal(ni);
        logPdfs = logPdfs.array() - logSumExp<float>(logPdfs);
        pdfs = logPdfs.array().exp();
        uint16_t zPrev = zi;
        zi = sampleDisc(pdfs, rnd);
        //      std::cout << z[i] << " " << Ksample << ": " << pdfs.transpose() << std::endl;
        if (zi == Ksample) {
          vmfsLock.lock();
          vmfs.push_back(base.posterior(ni,1).sample(rnd));
          vmfsLock.unlock();
          Ksample++;
        }
        if (zPrev != zi) {
          vmfSS[zPrev].topRows<3>() -= ni;
          vmfSS[zPrev](3) --;
          vmfSS[zi].topRows<3>() += ni;
          vmfSS[zi](3) ++;
          zCountS[i] = 1;
        } else {
          zCountS[i] ++;
        }
      }
      // sample dpvmf parameters
      {
        std::lock_guard<std::mutex> lock(vmfsLock);
        for (size_t k=0; k<Ksample; ++k) {
          if (vmfSS[k](3) > 0) {
            vmfs[k] = base.posterior(vmfSS[k]).sample(rnd);
          }
        }
      }
      {
        std::lock_guard<std::mutex> lockZs(zsLock);
        for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
          pl_w[i].z_ = zS[i];
        }
        K = Ksample;
      }

//      std::cout << "counts " << Ksample << ": ";
//      for (size_t k=0; k<Ksample; ++k) 
//        if (vmfSS[k](3) > 0) 
//          std::cout << vmfSS[k](3) << " ";
//      std::cout << "\ttaus: " ;
//      for (size_t k=0; k<Ksample; ++k) 
//        if (vmfSS[k](3) > 0) 
//          std::cout << vmfs[k].tau_ << " ";
//      std::cout << std::endl;
//      // sample points
//      for (int32_t i = 0; i!=iInsert;
//        i=(i+1)%nn.w_) {
//        tdp::Vector3fda& pi = pS[i];
//        tdp::Plane& pl = pl_w[i];
//        tdp::Vector5ida& ids = nn[i];
//
//        Eigen::Matrix3f SigmaPl;
//        Eigen::Matrix3f Info =  InfoO*pl.N_;
////        Eigen::Vector3f xi = SigmaO.ldlt().solve(pl.p_);
//        Eigen::Vector3f xi = Info*pl.p_; //*pl.w_;
//        for (int i=0; i<5; ++i) {
//          if (ids[i] > -1  && zS[ids[i]] < Ksample && tdp::IsValidData(pS[ids[i]])) {
//            SigmaPl = vmfs[zS[ids[i]]].mu_*vmfs[zS[ids[i]]].mu_.transpose();
//            Info += SigmaPl;
//            xi += SigmaPl*pS[ids[i]];
//          }
//        }
//        Eigen::Matrix3f Sigma = Info.inverse();
//        Eigen::Vector3f mu = Info.ldlt().solve(xi);
////        std::cout << xi.transpose() << " " << mu.transpose() << std::endl;
//        pi = Normal<float,3>(mu, Sigma).sample(rnd);
//      }
    };
  });

  float lambDPvMFmeans = cos(55.*M_PI/180.);
  tdp::DPvMFmeansSimple3fda dpvmf(lambDPvMFmeans);

  std::thread mapping([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
//    std::random_device rd_;
    std::mt19937 gen_(0);
    while(runMappingThread.Get()) {
    if (updateMap) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        iRead = nn.iRead_;
        iInsert = nn.iInsert_;
      }
      // compute gradient
      size_t numGrads = 0;
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        tdp::VectorkNNida& ids = nn.GetCircular(i);
        tdp::Plane& pl = pl_w.GetCircular(i);
        tdp::Vector3fda& Jn = Jn_w[i];
        tdp::Vector3fda& Jp = Jp_w[i];
        Jn = tdp::Vector3fda::Zero();
        Jp = tdp::Vector3fda::Zero();
        if (doRegvMF && lambdaRegDir > 0) {
          if (runSampling.Get()) {
            vmfsLock.lock();
            if (vmfs[pl.z_].tau_ > 0) {
              Jn = -lambdaRegDir*vmfs[pl.z_].mu_*vmfs[pl.z_].tau_;
            }
            vmfsLock.unlock();
          } else {
            dpvmfLock.lock();
            Jn = -lambdaRegDir*dpvmf.GetCenter(pl.z_);
            dpvmfLock.unlock();
          }
          if (!tdp::IsValidData(Jn)) {
            std::cout << Jn.transpose() << " " << pl.z_ <<
              vmfs[pl.z_].mu_.transpose() << ", " << vmfs[pl.z_].tau_ << std::endl;
            Jn = tdp::Vector3fda::Zero();
          }
        }
        if (doRegPc0) {
          Jp += -2.*lambdaRegPc0*(pc0_w[i] - pl.p_);
        }
        if (doRegAbsPc) {
          Jp += pl.n_ *2.*pl.n_.dot(numSum_w[i]*pl.p_ - pcSum_w[i]);
        }
        if (doRegAbsN) {
          Jn += 2*(numSum_w[i]*pl.n_ - nSum_w[i]);
        }
        for (int j=0; j<kNN; ++j) {
          if (ids[j] > -1){
            const tdp::Plane& plO = pl_w[ids[j]];
            if (doRegRelPlZ) {
              if (pl.z_ == plO.z_) {
                Jn +=  2.*lambdaRegPl*(pl.p2plDist(plO.p_))*(plO.p_-pl.p_);
                Jp += -2.*lambdaRegPl*(pl.p2plDist(plO.p_))*pl.n_;
              }
            }
            if (doRegRelNZ) {
              if (pl.z_ == plO.z_) {
                Jn +=  2.*lambdaRegPl*(pl.n_ - plO.n_);
              }
            }
            if (mapObsNum[i](j) > 0.) {
              if (doRegRelNObs) {
                Jn += mapObsNum[i](j)*2.*(pl.n_.dot(plO.n_)-mapObsDot[i](j)/mapObsNum[i](j))*plO.n_;
              }
              if (doRegRelPlObs) {
                Jn += mapObsNum[i](j)*2.*(pl.p2plDist(plO.p_)-mapObsP2Pl[i](j)/mapObsNum[i](j))*(plO.p_-pl.p_);
                Jp += -mapObsNum[i](j)*2.*(pl.p2plDist(plO.p_)-mapObsP2Pl[i](j)/mapObsNum[i](j))*pl.n_;
              }
              numGrads++;
            }
          }
        }
      }
//      std::cout << "have " << numGrads << " gradients > 0  off " << iInsert-iRead  << std::endl;
      // apply gradient
      for (int32_t i = 0; i!=iInsert;
        i=(i+1)%nn.w_) {
        tdp::Plane& pl = pl_w.GetCircular(i);
        tdp::Vector3fda& Jn = Jn_w[i];
        tdp::Vector3fda& Jp = Jp_w[i];
        std::lock_guard<std::mutex> mapGuard(mapLock);
        pl.n_ = (pl.n_- alphaGrad * Jn).normalized();
//        std::cout << Jn.transpose() << " " << pl.n_.transpose() << std::endl;
        pl.p_ -= alphaGrad * Jp;
        pc_w[i] = pl.p_;
        n_w[i] = pl.n_;
      }
//      std::cout << "map updated " << i << " " 
//        << (alphaGrad * Jn.transpose()) << "; "
//        << (alphaGrad * Jp.transpose()) << std::endl;
    }
    };
  });

  tdp::ProjectiveAssociation<CameraT::NumParams, CameraT> projAssoc(cam, w, h);

  std::vector<std::pair<size_t, size_t>> assoc;
  assoc.reserve(10000);

  uint32_t numObs = 0;
  uint32_t numInlPrev = 0;


  std::vector<std::vector<uint32_t>> invInd;
  std::vector<size_t> id_w;
  id_w.reserve(MAP_SIZE);

//  std::random_device rd;
  std::mt19937 gen(19023);

  mask.Fill(0);
  std::vector<uint32_t> idsCur;
  idsCur.reserve(w*h);

  std::ofstream out("trajectory_tumFormat.csv");
  out << "# " << input_uri << std::endl;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (runLoopClosure.GuiChanged()) {
      showLoopClose = runLoopClosure;
    }
    if (runLoopClosureGeom.GuiChanged()) {
      showLoopClose = runLoopClosureGeom;
    }
    if (pangolin::Pushed(icpReset)) {
      T_wc = tdp::SE3f();
    }

    if (!gui.paused() && !gui.finished()
        && frame > 0
        && (runMapping || frame == 1) 
        && (trackingGood || frame < 10)) { // add new observations
      TICK("mask");

      // update mask only once to know where to insert new planes
      TICK("data assoc");
      projAssoc.Associate(vbo_w, T_wc.Inverse(), dMin, dMax, 
          pl_w.SizeToRead());
      TOCK("data assoc");
      TICK("extract assoc");
      z.Fill(0);
      idsCur.clear();
      projAssoc.GetAssoc(z, mask, idsCur);
      std::random_shuffle(idsCur.begin(), idsCur.end());
      TOCK("extract assoc");

      tdp::GradientNormBiasedResampleEmptyPartsOfMask(pc, cam, mask,
          greyGradNorm, W, subsample, gen, 32, 32, w, h);
//      tdp::UniformResampleEmptyPartsOfMask(pc, cam, mask, W,
//          subsample, gen, 32, 32, w, h);
      TOCK("mask");
      {
        iReadCurW = pl_w.iInsert_;
        std::lock_guard<std::mutex> lock(pl_wLock); 
        TICK("normals");
        ExtractPlanes(pc, rgb, grey, greyFl, gradGrey,
             mask, W, frame, T_wc, cam, dpc, pl_w, pc_w, pc0_w, rgb_w,
            n_w, rs);
        TOCK("normals");
        TICK("add to model");
        for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
          dpvmf.addObservation(&pl_w[i].n_, &pl_w[i].z_);
          pcSum_w[i] = pl_w[i].p_;
          nSum_w[i] = pl_w[i].n_;
          numSum_w[i] ++;
        }
      }
//      vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
      vbo_w.Upload(&pc_w.ptr_[iReadCurW], 
          pc_w.SizeToRead(iReadCurW)*sizeof(tdp::Vector3fda), 
          iReadCurW*sizeof(tdp::Vector3fda));
      rbo.Upload(&rs.ptr_[iReadCurW],
          rs.SizeToRead(iReadCurW)*sizeof(float),
          iReadCurW*sizeof(float));

      TOCK("add to model");
      std::cout << " # map points: " << pl_w.SizeToRead() 
        << " " << dpvmf.GetZs().size() << std::endl;
      TICK("dpvmf");
      if (!runSampling.Get()) {
        dpvmfLock.lock();
        dpvmf.iterateToConvergence(100, 1e-6);
        dpvmfLock.unlock();
        K = dpvmf.GetK();
      }
      {
        std::lock_guard<std::mutex> lockZs(zsLock);
        for (size_t k=0; k<K; ++k) {
          if (k >= invInd.size()) {
            invInd.push_back(std::vector<uint32_t>());
            invInd.back().reserve(10000);
          } else {
            invInd[k].clear();
          }
        }
        if (pruneAssocByRender) {
          // only use ids that were found by projecting into the current pose
          for (auto i : idsCur) {
            uint32_t k = pl_w[i].z_;
            if (invInd[k].size() < 10000)
              invInd[k].push_back(i);
          }
        } else {      
          id_w.resize(pl_w.SizeToRead());
          std::iota(id_w.begin(), id_w.end(), 0);
          std::random_shuffle(id_w.begin(), id_w.end());
          // use all ids in the current map
          for (auto i : id_w) {
            uint32_t k = pl_w[i].z_;
            if (invInd[k].size() < 10000)
              invInd[k].push_back(i);
          }
        }
        uint32_t Kcur = K;
      }
      TOCK("dpvmf");
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();

    int64_t t_host_us_d = 0;
    TICK("Setup");
    if (gui.verbose) std::cout << "collect d" << std::endl;
    rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
    if (gui.verbose) std::cout << "compute pc" << std::endl;
    rig.ComputePc(cuD, true, pcs_c);
    pc.CopyFrom(pcs_c.GetImage(0));
    if (gui.verbose) std::cout << "collect rgb" << std::endl;
    rig.CollectRGB(gui, rgb) ;
    cuRgb.CopyFrom(rgb);
    if (gui.verbose) std::cout << "compute grey" << std::endl;
    tdp::Rgb2Grey(cuRgb,cuGreyFl,1./255.);
    tdp::Gradient(cuGreyFl, cuGreyDu, cuGreyDv, cuGradGrey);
    tdp::Gradient2AngleNorm(cuGreyDu, cuGreyDv,
      cuGreyGradTheta, cuGreyGradNorm);
    greyGradNorm.CopyFrom(cuGreyGradNorm);
//    cuGreyFlSmooth.CopyFrom(cuGreyFl);
////    tdp::Blur5(cuGreyFl,cuGreyFlSmooth, 10.);
//    tdp::Convert(cuGreyFlSmooth, cuGrey, 255.);
    greyFl.CopyFrom(cuGreyFl);
//    greyFl.CopyFrom(cuGreyFlSmooth);
//    tdp::Gradient(cuGreyFlSmooth, cuGreyDu, cuGreyDv, cuGradGrey);
    gradGrey.CopyFrom(cuGradGrey);

    n.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    size_t numProjected =0;
    trackingGood = false;
    if (frame > 1 && runTracking && !gui.finished()) { // tracking
      TICK("icp");
      Eigen::Matrix<float,6,6> A;
      Eigen::Matrix<float,6,1> b;
      Eigen::Matrix<float,6,1> Ai;
      float dotThr = cos(angleThr*M_PI/180.);
      std::vector<size_t> indK(invInd.size(),0);
      for (size_t it = 0; it < maxIt; ++it) {
          mask.Fill(0);
          assoc.clear();
//          pc_c.MarkRead();
//          n_c.MarkRead();
          indK = std::vector<size_t>(invInd.size(),0);
          numProjected = 0;

        A = Eigen::Matrix<float,6,6>::Zero();
        b = Eigen::Matrix<float,6,1>::Zero();
        Ai = Eigen::Matrix<float,6,1>::Zero();
        float err = 0.;
        float H = 1e10;
        float Hprev = 1e10;
        uint32_t numInl = 0;
        numObs = 0;

        tdp::SE3f T_cw = T_wc.Inverse();
//        size_t numInl0 = numInl;
        numInlPrev = numInl;
        // associate new data until enough
        bool exploredAll = false;
        uint32_t k = 0;
        while (numObs < 10000 && !exploredAll) {
          k = (k+1) % invInd.size();
          while (indK[k] < invInd[k].size()) {
            size_t i = invInd[k][indK[k]++];
            tdp::Plane& pl = pl_w.GetCircular(i);
            numProjected++;
            int32_t u, v;
            if (angleThr > 0.) {
              if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, cam, pc,
                    W, dpc, n, curv, u,v ))
                continue;
              if (useTexture) {
                if (!AccumulateP2Pl(pl, T_wc, T_cw, cam, pc(u,v), n(u,v), 
                      greyFl(u,v), distThr, p2plThr, dotThr, lambdaTex,
                      A, Ai, b, err))
                  continue;
              } else if (useNormals) {
                if (!AccumulateP2Pl(pl, T_wc, T_cw, cam, pc(u,v), n(u,v), 
                      greyFl(u,v), distThr, p2plThr, dotThr, lambdaNs, lambdaTex,
                      A, Ai, b, err)) {
                  continue;
                }
              } else {
                if (!AccumulateP2Pl(pl, T_wc, T_cw, pc(u,v), n(u,v),
                      distThr, p2plThr, dotThr, A, Ai, b, err))
                  continue;
              }
            } else {
              if (!tdp::ProjectiveAssoc(pl, T_cw, cam, pc, u,v ))
                continue;
              if (!AccumulateP2Pl(pl, T_wc, T_cw, pc(u,v), 
                    distThr, p2plThr, A, Ai, b, err))
                continue;
            }
            pl.lastFrame_ = frame;
            pl.numObs_ ++;
            numInl ++;
            mask(u,v) ++;
            assoc.emplace_back(i,u+v*pc.w_);
            break;
          }

          if (numInl > numInlPrev
              && k == 0) {
            if (tdp::CheckEntropyTermination(A, Hprev, HThr, condEntropyThr, 
                  negLogEvThr, H))
              break;
            Hprev = H;
            numObs ++;
            numInlPrev = numInl;
          }

          exploredAll = true;
          for (size_t k=0; k<indK.size(); ++k) {
            exploredAll &= indK[k] >= invInd[k].size();
          }
        }
//        std::cout << " added " << numInl - numInl0 << std::endl;
        Eigen::Matrix<float,6,1> x = Eigen::Matrix<float,6,1>::Zero();
        if (numInl > 10) {
          // solve for x using ldlt
          x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
          T_wc = T_wc * tdp::SE3f::Exp_(x);
        }
        if (gui.verbose) {
          std::cout << "\tit " << it << ": err=" << err 
            << "\t# inliers: " << numInl
            << "\t|x|: " << x.topRows(3).norm()*180./M_PI 
            << " " <<  x.bottomRows(3).norm()
            << std::endl;
        }
        if (x.topRows<3>().norm()*180./M_PI < icpdRThr
            && x.bottomRows<3>().norm() < icpdtThr
            && tdp::CheckEntropyTermination(A, Hprev, HThr, 0.f, negLogEvThr, H)) {
          std::cout << numInl << " " << numObs << " " << numProjected << std::endl;
          break;
        }
      }
      for (size_t k=0; k<indK.size(); ++k) {
        if (invInd[k].size() > 0 )
          std::cout << "used different directions " << k << "/" 
            << invInd.size() << ": " << indK[k] 
            << " of " << invInd[k].size() << std::endl;
      }
      logObs.Log(log(numObs)/log(10.), log(numInlPrev)/log(10.), 
          log(numProjected)/log(10.), log(pl_w.SizeToRead())/log(10));
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
      Eigen::Matrix<float,6,1> ev = eig.eigenvalues().real();
      Eigen::Matrix<float,6,6> Q = eig.eigenvectors();
      float H = -ev.array().log().sum();
      std::cout << " H " << H << " neg log evs " << 
        -ev.array().log().matrix().transpose() << std::endl;

//      for (size_t k=0; k<K; ++k) {
//        Eigen::Matrix<float,6,1> Ai;
//        Ai << Eigen::Vector3f::Zero(), dpvmf.GetCenter(k);
//        std::cout << "k " << k << std::endl;
//        std::cout << (Q.transpose()*Ai*Ai.transpose()*Q).diagonal().transpose() << std::endl;
//      }

      logEntropy.Log(H);
      logEig.Log(-ev.array().log().matrix());
      Eigen::Matrix<float,6,1> q0 = Q.col(0);
      uint32_t maxId = 0;
      q0.array().abs().maxCoeff(&maxId);
      q0 *= (q0(maxId) > 0? 1.: -1.);
      logEv.Log(q0);
      T_wcs.push_back(T_wc);
      trackingGood = H <= HThr && numInlPrev > 10;
      TOCK("icp");
      if (trackingGood) {
        std::cout << "tracking good" << std::endl;
      }

      if (updatePlanes && trackingGood) {
        std::lock_guard<std::mutex> mapGuard(mapLock);
        TICK("update planes");
        size_t numNN = 0;
//        tdp::SE3f T_cw = T_wc.Inverse();
        for (const auto& ass : assoc) {

          int32_t u = ass.second%pc.w_;
          int32_t v = ass.second/pc.w_;
//          int32_t uRe = ass.second%pc.w_;
//          int32_t vRe = ass.second/pc.w_;
//          tdp::Plane& pl = pl_w[ass.first];
//          if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, cam, pc,
//                W, dpc, n, curv, uRe,vRe ))
//            continue;
//          std::cout << u << " " << v << " redone: " << u << " " << v << std::endl;

          tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);
          tdp::Vector3fda n_c_in_w = T_wc.rotation()*n(u,v);
          if (!updateMap) {
            pl_w[ass.first].AddObs(pc_c_in_w, n_c_in_w);
            n_w[ass.first] =  pl_w[ass.first].n_;
            pc_w[ass.first] = pl_w[ass.first].p_;
          } 
          float w = numSum_w[ass.first];
          pcSum_w[ass.first] += pc_c_in_w;
          nSum_w[ass.first] += n_c_in_w;
          numSum_w[ass.first] ++;
          for (size_t i=0; i<kNN; ++ i) {
            for (const auto& assB : assoc) {
              if (assB.first == nn[ass.first](i)){
                int32_t uB = assB.second%pc.w_;
                int32_t vB = assB.second/pc.w_;
//                mapObsP2Pl[ass.first](i) += n(u,v).dot(pc(uB,vB)-pc(u,v));
//                mapObsDot[ass.first](i) += n(u,v).dot(n(uB,vB));
//                mapObsNum[ass.first](i) ++;
                float w = mapObsNum[ass.first](i);
                mapObsP2Pl[ass.first](i) = (w*mapObsP2Pl[ass.first](i) + n(u,v).dot(pc(uB,vB)-pc(u,v)))/(w+1.);
                mapObsDot[ass.first](i) = (mapObsDot[ass.first](i)*w + n(u,v).dot(n(uB,vB)))/(w+1.);
                mapObsNum[ass.first](i) = std::min(100.f,w+1.f) ;
//                std::cout << "found NN " << i << " of " << ass.first 
//                  << " " << mapObsP2Pl[ass.first](i)/mapObsNum[ass.first](i) 
//                  << " " << mapObsDot[ass.first](i)/mapObsNum[ass.first](i)
//                  << " " << mapObsNum[ass.first](i) << std::endl;
                numNN++;
                break;
              }
            }
          }
        }
        std::cout << "num NN measured " << numNN << std::endl;
        TOCK("update planes");
      }
    }

    if (runLoopClosureGeom && K>2) {
      tdp::ManagedDPvMFmeansSimple3fda dpvmfCur(lambDPvMFmeans);
      for (const auto& ass : assoc) {
        dpvmfCur.addObservation(n(ass.second%pc.w_,ass.second/pc.w_));
      }
      dpvmfCur.iterateToConvergence(100, 1e-6);
      if (dpvmfCur.GetK() > 2) {
        std::vector<size_t> idsW(K);
        std::vector<size_t> idsC(dpvmfCur.GetK());
        std::iota(idsW.begin(), idsW.end(), 0);
        std::iota(idsC.begin(), idsC.end(), 0);
        Eigen::Matrix3f N;
        float maxAlign = 0;
        for (size_t it =0; it < 1000; ++it) {
          std::random_shuffle(idsW.begin(), idsW.end());
          std::random_shuffle(idsC.begin(), idsC.end());
          N = Eigen::Matrix3f::Zero();
          for (size_t i=0; i<3; ++i) {
            N += dpvmf.GetCenter(idsW[i]) * dpvmfCur.GetCenter(idsC[i]).transpose();
          }
          // TODO check order
          Eigen::Matrix3f R_wc = tdp::ProjectOntoSO3<float>(N);
          float align = (R_wc*N).trace();
          if (align > maxAlign) {
            T_wcRansac.rotation() = tdp::SO3f(R_wc);
          }
        }
      }
    }

    frame ++;

    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    TICK("Draw 3D");

    if (showPcCurrent) {
      vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
    }

    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      glColor4f(0.,1.,0.,1.0);
//      pangolin::glDrawAxis(T_wc.matrix(), 0.05f);
      pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wc.matrix(), 0.1f);

      if (showLoopClose) {
        glColor4f(1.,0.,0.,1.0);
        //      pangolin::glDrawAxis(T_wcRansac.matrix(), 0.05f);
        pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wcRansac.matrix(), 0.1f);
      }
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,20, 0.03f);

      if (showSamples) {
        nbo_w.Upload(nS.ptr_, nS.SizeBytes(), 0);
      } else {
        nbo_w.Upload(n_w.ptr_, n_w.SizeBytes(), 0);
      }

      if (showFullPc) {
        // TODO I should not need to upload all of pc_w everytime;
        // might break things though
        vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
        cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeBytes(), 0);
//        cbo_w.Upload(&rgb_w.ptr_[iReadCurW], 
//            rgb_w.SizeToRead(iReadCurW)*sizeof(tdp::Vector3fda), 
//            iReadCurW*sizeof(tdp::Vector3fda));
        pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
        pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
        if (showAge || showObs || showCurv) {
          age.Reinitialise(pl_w.SizeToRead());
          if (showAge) {
            for (size_t i=0; i<age.Area(); ++i) 
              age[i] = pl_w.GetCircular(i).lastFrame_;
          } else if (showObs) {
            for (size_t i=0; i<age.Area(); ++i) 
              age[i] = pl_w.GetCircular(i).numObs_;
          } else {
            for (size_t i=0; i<age.Area(); ++i) 
              age[i] = pl_w.GetCircular(i).curvature_;
          }
          valuebo.Reinitialise(pangolin::GlArrayBuffer, age.Area(),  GL_FLOAT,
              1, GL_DYNAMIC_DRAW);
          valuebo.Upload(age.ptr_,  age.SizeBytes(), 0);

          std::pair<float,float> minMaxAge = age.MinMax();
          tdp::RenderVboValuebo(vbo_w, valuebo, minMaxAge.first, minMaxAge.second,
              P, MV);
        } else if (showLabels && frame > 1) {
          labels.Reinitialise(pl_w.SizeToRead());
          if (showDPvMFlabels) {
            for (size_t i=0; i<labels.Area(); ++i) 
              labels[i] = zS[i] == 9999? 0 : zS[i];
          } else {
            for (size_t i=0; i<labels.Area(); ++i) 
              labels[i] = pl_w.GetCircular(i).z_;
          }
          lbo.Reinitialise(pangolin::GlArrayBuffer, labels.Area(),  GL_UNSIGNED_SHORT,
              1, GL_DYNAMIC_DRAW);
          lbo.Upload(labels.ptr_,  labels.SizeBytes(), 0);
          std::pair<float,float> minMaxAge = labels.MinMax();
          std::cout << "render labels " << minMaxAge.first << " " << minMaxAge.second << std::endl;
          tdp::RenderLabeledVbo(vbo_w, lbo, s_cam, minMaxAge.first,
              minMaxAge.second);
          std::cout << "render labels done" << std::endl;
        } else if (showSurfels) {
//          std::cout << "rbo upload " << rs.SizeBytes() << std::endl;
//          rbo.Upload(rs.ptr_, rs.SizeBytes(), 0);
          std::cout << "render surfels" << std::endl;
          tdp::RenderSurfels(vbo_w, nbo_w, cbo_w, rbo, dMax, P, MV);
          std::cout << "rendered surfels" << std::endl;
        } else {
          pangolin::RenderVboCbo(vbo_w, cbo_w, true);
        }
        if (showNN) {
//          std::cout << pl_w.SizeToRead() << " vs " << mapNN.size() << " -> "
//             << mapNN.size()/kNN << std::endl;
          glColor4f(0.3,0.3,0.3,0.3);
          for (auto& ass : mapNN) {
            if (ass.second >= 0)
              tdp::glDrawLine(pl_w[ass.first].p_, pl_w[ass.second].p_);
          }
        }
      }

      if (showNormals) {
        std::cout << "render normals local" << std::endl;
        tdp::ShowCurrentNormals(pc, n, assoc, T_wc, scale);
        std::cout << "render normals global " << n_w.SizeToRead() << std::endl;
        pangolin::glUnsetFrameOfReference();
        glColor4f(0,1,0,0.5);
        for (size_t i=0; i<n_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w.GetCircular(i), 
              pc_w.GetCircular(i) + scale*n_w.GetCircular(i));
        }
        std::cout << "render normals done" << std::endl;
      }

      // render current camera second in the propper frame of
      // reference
      if (showPcCurrent) {
        pangolin::glSetFrameOfReference(T_wc.matrix());
        if(dispLvl == 0){
          pangolin::RenderVboCbo(vbo, cbo, true);
        } else {
          glColor3f(1,0,0);
          pangolin::RenderVbo(vbo);
        }
        pangolin::glUnsetFrameOfReference();
      }
    }

    if (viewAssoc.IsShown()) {
      viewAssoc.Activate(s_cam);
      pangolin::glSetFrameOfReference(T_wc.matrix());
      pangolin::glDrawAxis(0.1f);
      if (showPcCurrent) {
        pangolin::RenderVboCbo(vbo, cbo, true);
      }
      pangolin::glUnsetFrameOfReference();

      pangolin::glDrawAxis(0.3f);
      glColor4f(1,0,0,1.);
      for (const auto& ass : assoc) {
        tdp::Vector3fda pc_c_in_m = T_wc*pc(ass.second%pc.w_,ass.second/pc.w_);
        tdp::glDrawLine(pl_w[ass.first].p_, pc_c_in_m);
      }
    }

    if (viewNormals.IsShown()) {
      Eigen::Matrix4f Tview = s_cam.GetModelViewMatrix();
      Tview(0,3) = 0.; Tview(1,3) = 0.; Tview(2,3) = -2.2;
      normalsCam.GetModelViewMatrix() = Tview;
      viewNormals.Activate(normalsCam);
      glColor4f(0,0,1,0.5);
//      nbo_w.Upload(n_w.ptr_, n_w.SizeBytes(), 0);
      pangolin::RenderVbo(nbo_w);
      glColor4f(1,0,0,1.);
      for (size_t k=0; k<dpvmf.GetK(); ++k) {
        tdp::glDrawLine(tdp::Vector3fda::Zero(), dpvmf.GetCenter(k));
      }
      glColor4f(0,1,0,1.);
      {
        std::lock_guard<std::mutex> lock(vmfsLock);
        for (size_t k=0; k<vmfs.size(); ++k) {
          if (vmfSS[k](3) > 0)
            tdp::glDrawLine(tdp::Vector3fda::Zero(), vmfs[k].mu_);
        }
      }
    }

    TOCK("Draw 3D");
    if (gui.verbose) std::cout << "draw 2D" << std::endl;
    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);
    if (viewGreyGradNorm.IsShown()) {
      viewGreyGradNorm.SetImage(greyGradNorm);
    }
    if (viewCurrent.IsShown()) {
      viewCurrent.SetImage(rgb);
      glColor3f(1,0,0);
      for (size_t u=0; u<rgb.w_; ++u)
        for (size_t v=0; v<rgb.h_; ++v) {
          if (mask(u,v)) {
            pangolin::glDrawCircle(u,v,1);
          }
        }
      std::cout << "dots of mask done"<< std::endl;
    }

    if (containerTracking.IsShown()) {
      if (viewMask.IsShown()) {
        viewMask.SetImage(mask);
      }
      if (viewGrey.IsShown()) {
        viewGrey.SetImage(grey);
      }
      std::cout << "grey image and mask done"<< std::endl;
    }
    if (!gui.finished() && plotters.IsShown()) {
      std::cout << "scroll plots" << std::endl;
      plotdH.ScrollView(1,0);
      plotH.ScrollView(1,0);
      plotObs.ScrollView(1,0);
      plotEig.ScrollView(1,0);
      plotEv.ScrollView(1,0);
      std::cout << "scrolling done"<< std::endl;
    }

    TOCK("Draw 2D");
    std::cout << "2D done"<< std::endl;
    if (record) {
      std::string name = tdp::MakeUniqueFilename("sparseFusion.png");
      name = std::string(name.begin(), name.end()-4);
      gui.container().SaveOnRender(name);
    }
    std::cout << "record done"<< std::endl;

    if (gui.verbose) std::cout << "finished one iteration" << std::endl;
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  
    if (streamTimeStamps.size() > frame-1) {
       out << streamTimeStamps[frame-1] << " ";
    } else {
      out << pangolin::Time_us(pangolin::TimeNow())/1000000 << "."
        << pangolin::Time_us(pangolin::TimeNow())%1000000 << " ";
    }
    out << T_wc.translation()(0) << " "  // tx
      << T_wc.translation()(1) << " "  // ty
      << T_wc.translation()(2) << " "  // tz
      << T_wc.rotation().vector()(0) << " "  // qx
      << T_wc.rotation().vector()(1) << " "  // qy
      << T_wc.rotation().vector()(2) << " "  // qz
      << T_wc.rotation().vector()(3) << std::endl;  // qw
  }
  out.close();

//  imuInterp.Stop();
//  if (imu) imu->Stop();
//  delete imu;
//  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

