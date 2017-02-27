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
#include <tdp/icp/photoSO3.h>
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
#include <tdp/io/tinyply.h>

#include <tdp/sampling/sample.hpp>
#include <tdp/sampling/vmf.hpp>
#include <tdp/sampling/vmfPrior.hpp>
#include <tdp/sampling/normal.hpp>

//#include "planeHelpers.h"
//#include "icpHelper.h"
#include "visHelper.h"

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

#define PYR 4
#define kNN 12
#define MAP_SIZE 1000000

namespace tdp {

typedef Eigen::Matrix<float,kNN,1,Eigen::DontAlign> VectorkNNfda;
typedef Eigen::Matrix<int32_t,kNN,1,Eigen::DontAlign> VectorkNNida;

//https://ai2-s2-pdfs.s3.amazonaws.com/a8a6/18363b8dee8037df9133668ec8dcd532ee4e.pdf
void NoiseModelNguyen(
    const tdp::Vector3fda& n,
    const tdp::Vector3fda& p,
    const CameraT& cam,
    tdp::Matrix3fda& SigmaO
    ) {
  float theta = acos(std::min(1.f,std::max(-1.f,(float)n.dot(p)/p.norm())));
  float d_c = p(2);
  float sigmaL = (0.8f + 0.035*theta/(M_PI*0.5-theta))*d_c/cam.params_(0);
  float sigmaA = 0.;
  if (10.*M_PI/180. < theta && theta < 60.*M_PI/180.) {
    sigmaA = 0.0012 + 0.0019*(d_c-0.4)*(d_c-0.4);
  } else {
    sigmaA = 0.0012 + 0.0019*(d_c-0.4)*(d_c-0.4) + 0.0001/sqrtf(d_c)*(theta*theta/((M_PI*0.5-theta)*(M_PI*0.5-theta)));
  }
//  std::cout << sigmaL << ", " << sigmaA << std::endl;
  SigmaO = tdp::Matrix3fda::Zero();
  SigmaO(0,0) = sigmaL*sigmaL;
  SigmaO(1,1) = sigmaL*sigmaL;
  SigmaO(2,2) = sigmaA*sigmaA;
}

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
    ManagedHostCircularBuffer<Matrix3fda>& pc0Info_w,
    ManagedHostCircularBuffer<Vector3bda>& rgb_w,
    ManagedHostCircularBuffer<Vector3fda>& n_w,
    ManagedHostCircularBuffer<Vector3fda>& grad_w,
    ManagedHostCircularBuffer<float>& rs,
    ManagedHostCircularBuffer<uint16_t>& ts,
    bool viaRMLs
    ) {
  Plane pl;
  tdp::Brief feat;
  Vector3fda n, p;
  float curv;
  for (size_t i=0; i<mask.Area(); ++i) {
    if (mask[i]) {
//      std::cout << "mask point " << i << std::endl;
      if( tdp::IsValidData(pc[i]) ) {
//      uint32_t Wscaled = floor(W*pc[i](2));
      uint32_t Wscaled = W;
      const uint32_t u = i%mask.w_;
      const uint32_t v = i/mask.w_;
//      std::cout << "found valid point in mask " << u << "," << v << std::endl;
//      if (tdp::NormalViaScatter(pc, i%mask.w_, i/mask.w_, Wscaled, n)) {
      bool success = false;
      if (viaRMLs) {
        success = tdp::NormalViaRMLS(pc, u, v, Wscaled, 0.29, dpc, n, curv, p);
      } else {
        success = tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, dpc, n, curv, p);
      }
      if (success) {
//        std::cout << "extracted normal at " << u << "," << v << std::endl;
//        ExtractClosestBrief(pc, grey, pts, orientation, 
//            p, n, T_wc, cam, Wscaled, i%mask.w_, i/mask.w_, feat);
        pl.p_ = T_wc*p;
        pl.n_ = T_wc.rotation()*n;
        pl.curvature_ = curv;
        pl.rgb_ = rgb[i];
        pl.gradGrey_ = gradGrey[i];
        pl.gradNorm_ = pl.gradGrey_.norm();
        pl.grey_ = greyFl[i];
        pl.lastFrame_ = frame;
        pl.w_ = 1.;
        pl.numObs_ = 1;
        pl.valid_ = true;
        pl.Hn_ = 0;
        pl.Hp_ = 0;
//        pl.feat_ = feat;
//        pl.r_ = 2*W*pc[i](2)/cam.params_(0); // unprojected radius in m
        pl.r_ = p(2)/cam.params_(0); // unprojected radius in m

        pl.grad_ = pl.Compute3DGradient(T_wc, cam, u, v, gradGrey[i]);
//        float uGrad = u + 10.*pl.gradGrey_(0);
//        float vGrad = v + 10.*pl.gradGrey_(1);
//        tdp::Rayfda ray(tdp::Vector3fda::Zero(),
//            cam.Unproject(uGrad,vGrad,1.));
//        ray.Transform(T_wc);
//        pl.grad_ = pl.gradGrey_.norm()*(ray.IntersectPlane(pl.p_,pl.n_) - pl.p_).normalized();
        // could project onto plane spanned by normal?

        tdp::Matrix3fda SigmaO;
        NoiseModelNguyen(n, p, cam, SigmaO);
        SigmaO = T_wc.rotation().matrix().transpose()*SigmaO*T_wc.rotation().matrix().transpose();
        pc0Info_w[pl_w.iInsert_] = SigmaO.inverse();
        pl_w.Insert(pl);
        pc_w.Insert(pl.p_);
        pc0_w.Insert(pl.p_);
        n_w.Insert(pl.n_);
        grad_w.Insert(pl.grad_);
        rgb_w.Insert(pl.rgb_);
        rs.Insert(pl.r_);
        ts.Insert(pl.lastFrame_);
      }
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
    int32_t v,
    bool viaRMLs
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
//        if(tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, dpc, ni, curvi, pi)) {
        bool success = false;
        if (viaRMLs) {
          success = tdp::NormalViaRMLS(pc, u, v, Wscaled, 0.29, 
              dpc, ni, curvi, pi);
        } else {
          success = tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, 
              dpc, ni, curvi, pi);
        }
        if (success) {
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
    int32_t& v,
    bool normalViaRMLS
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  Eigen::Vector2f x = cam.Project(T_cw*pc_w);
  u = floor(x(0)+0.5f);
  v = floor(x(1)+0.5f);
  return EnsureNormal(pc, dpc, W, n, curv, u, v, normalViaRMLS);
}


bool AccumulateP2Pl(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
  if (n_w_in_c.dot(n_ci) > dotThr) {
    float p2pl = n_w.dot(pc_w - T_wc*pc_ci);
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

/// uses texture as well
bool AccumulateP2Pl3DGrad(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
    float u, float v,
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
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        // p2pl
        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() = n_w_in_c; 
//        Ai.bottomRows<3>() = n_w; 
        bi = p2pl;
        A += Ai * Ai.transpose();
        b += Ai * bi;
        err += bi;
//        std::cout << " p2pl " << bi << " " << Ai.transpose() << std::endl;
        tdp::Rayfda ray(tdp::Vector3fda::Zero(),
            cam.Unproject(u + gradGrey_ci(0),v + gradGrey_ci(1),1.));
        tdp::Vector3fda grad3d = ray.IntersectPlane(pc_ci,n_ci)-pc_ci;
        // texture inverse transform verified Jse3 
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << SO3mat<float>::invVee(T_cw.rotation()*(pc_w-T_wc.translation())), 
             -Eigen::Matrix3f::Identity();
        Ai = Jse3.transpose() * grad3d;
        bi = - grey_ci + pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);
//        std::cout << " intensity " << bi << " " << Ai.transpose() << std::endl;
//        std::cout << Jse3 << std::endl;
//        std::cout << Jpi << std::endl;
//        std::cout << gradGrey_ci << std::endl;
        err += lambda*bi;
        // accumulate
        return true;
      }
  }
  return false;
}


/// uses texture as well
bool AccumulateP2PlIntensity(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
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
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - pc_c_in_w);
      if (fabs(p2pl) < p2plThr) {
        // p2pl
        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>() = n_w_in_c; 
//        Ai.bottomRows<3>() = n_w; 
        bi = p2pl;
        A += Ai * Ai.transpose();
        b += Ai * bi;
        err += bi;
//        std::cout << " p2pl " << bi << " " << Ai.transpose() << std::endl;
        // texture inverse transform verified Jse3 
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(T_cw*pc_w);
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << SO3mat<float>::invVee(T_cw.rotation()*(pc_w-T_wc.translation())), 
             -Eigen::Matrix3f::Identity();
        Ai = Jse3.transpose() * Jpi.transpose() * gradGrey_ci;
        bi = - grey_ci + pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);

//        std::cout << " intensity " << bi << " " << Ai.transpose() << std::endl;
//        std::cout << Jse3 << std::endl;
//        std::cout << Jpi << std::endl;
//        std::cout << gradGrey_ci << std::endl;
        err += lambda*bi;
        // accumulate
        return true;
      }
  }
  return false;
}


/// uses texture and normal as well
/// the use of the image gradient information was wrong here since this
/// derivation was using the fact that we know the pose between
/// observing the gradiant and the current image. This is not given
/// since we only know pose to world;
//bool AccumulateP2Pl(const Plane& pl, 
//    tdp::SE3f& T_wc, 
//    tdp::SE3f& T_cw, 
//    CameraT& cam,
//    const Vector3fda& pc_ci,
//    const Vector3fda& n_ci,
//    float grey_ci,
//    float distThr, 
//    float p2plThr, 
//    float dotThr,
//    float gamma,
//    float lambda,
//    Eigen::Matrix<float,6,6>& A,
//    Eigen::Matrix<float,6,1>& Ai,
//    Eigen::Matrix<float,6,1>& b,
//    float& err
//    ) {
//  const tdp::Vector3fda& n_w =  pl.n_;
//  const tdp::Vector3fda& pc_w = pl.p_;
//  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
//  float bi=0;
//  float dist = (pc_w - pc_c_in_w).norm();
//  if (dist < distThr) {
//    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
//    if (n_w_in_c.dot(n_ci) > dotThr) {
//      float p2pl = n_w.dot(pc_w - pc_c_in_w);
//      if (fabs(p2pl) < p2plThr) {
//        // p2pl
//        Ai.topRows<3>() = pc_ci.cross(n_w_in_c); 
//        Ai.bottomRows<3>() = n_w_in_c; 
//        bi = p2pl;
//        A += Ai * Ai.transpose();
//        b += Ai * bi;
//        err += bi;
////        std::cout << "--" << std::endl;
////        std::cout << Ai.transpose() << "; " << bi << std::endl;
//        // normal old
////        Ai.topRows<3>() = -n_ci.cross(n_w_in_c); 
////        Ai.bottomRows<3>().fill(0.); 
////        bi = n_ci.dot(n_w_in_c) - 1.;
////        A += gamma*(Ai * Ai.transpose());
////        b += gamma*(Ai * bi);
////        err += gamma*bi;
//        // normal new
//        Eigen::Matrix3f Asi = -T_wc.rotation().matrix()*tdp::SO3fda::invVee(n_ci);
//        Eigen::Vector3f bsi = -(T_wc.rotation()*n_ci - n_w);
//        A.topLeftCorner<3,3>() += gamma*(Asi*Asi.transpose());
//        b.topRows<3>() += gamma*(Ai.transpose() * bi);
//        err += gamma*bi.norm();
////        std::cout << Ai.transpose() << "; " << bi << std::endl;
//        // texture
//        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_c_in_w);
//        Eigen::Matrix<float,3,6> Jse3;
//        Jse3.leftColumns<3>() = -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci));
//        Jse3.rightColumns<3>() = Eigen::Matrix3f::Identity();
////        Jse3 << -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci)), 
////             Eigen::Matrix3f::Identity();
//        // TODO: should not be using the model image gradient here!!
//        Ai = Jse3.transpose() * Jpi.transpose() * pl.gradGrey_;
//        bi = grey_ci - pl.grey_;
//        A += lambda*(Ai * Ai.transpose());
//        b += lambda*(Ai * bi);
//        err += lambda*bi;
////        std::cout << Ai.transpose() << "; " << bi << std::endl;
//        // accumulate
//        return true;
//      }
//    }
//  }
//  return false;
//}
//
/// uses normals and p2pl
bool AccumulateP2PlNormal(const Plane& pl, 
    const tdp::Vector3fda& n_w,
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float p2plThr, 
    float dotThr,
    float gamma,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& pc_w = pl.p_;
  float bi=0;
    Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
    if (n_w_in_c.dot(n_ci) > dotThr) {
      float p2pl = n_w.dot(pc_w - T_wc*pc_ci);
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
        // normal old
        Ai.topRows<3>() = -n_ci.cross(n_w_in_c); 
        Ai.bottomRows<3>().fill(0.); 
        bi = n_ci.dot(n_w_in_c) - 1.;
        A += gamma*(Ai * Ai.transpose());
        b += gamma*(Ai * bi);
        err += gamma*bi;
        // normal new
//        Eigen::Matrix3f Asi = -T_wc.rotation().matrix()*tdp::SO3fda::invVee(n_ci);
//        Eigen::Vector3f bsi = -(T_wc.rotation()*n_ci - n_w);
//        A.topLeftCorner<3,3>() += gamma*(Asi*Asi.transpose());
//        b.topRows<3>() += gamma*(Asi.transpose() * bsi);
//        err += gamma*bsi.norm();
        // accumulate
        return true;
      }
  }
  return false;
}

/// uses gradient only
bool AccumulateIntDiff(const Plane& pl, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
    float lambda,
    Eigen::Matrix<float,3,3>& A,
    Eigen::Matrix<float,3,1>& Ai,
    Eigen::Matrix<float,3,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_w_in_c = T_cw*pc_w;
        // texture
//        if (tdp::IsValidData(pl.grad_)) {
//          Eigen::Matrix<float,3,6> Jse3;
//          Jse3.leftCols<3>() = -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci));
//          Jse3.rightCols<3>() = Eigen::Matrix3f::Identity();
//          Ai = Jse3.transpose() * pl.grad_;
//          bi = grey_ci - pl.grey_;
//          A += lambda*(Ai * Ai.transpose());
//          b += lambda*(Ai * bi);
//          err += lambda*bi;
//        } else {
//          std::cout << "gradient is nan " << std::endl;
//        }
        // texture inverse transform verified Jse3 
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_w_in_c);
        Eigen::Matrix<float,3,3> Jso3 = SO3mat<float>::invVee(T_cw.rotation()*pc_w);
        Ai = Jso3.transpose() * Jpi.transpose() * gradGrey_ci;
        float bi = -grey_ci + pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);
        err += lambda*bi;
//        std::cout << " intensity SO3 " << bi << " " << Ai.transpose() << std::endl;
        // accumulate
        return true;
}

/// uses gradient only
bool AccumulatePhotoSO3only(
    tdp::SO3f& R_cp, 
    CameraT& cam,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
    tdp::Vector3fda dir_p,
    float grey_pi, // previous image
    Eigen::Matrix<float,3,3>& A,
    Eigen::Matrix<float,3,1>& Ai,
    Eigen::Matrix<float,3,1>& b,
    float& err
    ) {
  // texture inverse transform verified Jse3 
  Eigen::Matrix<float,2,3> Jpi = cam.Jproject(R_cp*dir_p);
  Eigen::Matrix<float,3,3> Jso3 = -R_cp.matrix()*SO3mat<float>::invVee(dir_p);
  Ai = Jso3.transpose() * Jpi.transpose() * gradGrey_ci;
  float bi = -grey_ci + grey_pi;
  A += Ai*Ai.transpose();
  b += Ai*bi;
  err += bi;
  return true;
}

/// uses gradient and normal as well
bool AccumulateP2PlIntensityNormals(const Plane& pl, 
    const tdp::Vector3fda& n_w,
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
    float p2plThr, 
    float dotThr,
    float gamma,
    float lambda,
    Eigen::Matrix<float,6,6>& A,
    Eigen::Matrix<float,6,1>& Ai,
    Eigen::Matrix<float,6,1>& b,
    float& err
    ) {
  const tdp::Vector3fda& pc_w = pl.p_;
  tdp::Vector3fda pc_c_in_w = T_wc*pc_ci;
  float bi=0;
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
        // normal old
//        Ai.topRows<3>() = -n_ci.cross(n_w_in_c); 
//        Ai.bottomRows<3>().fill(0.); 
//        bi = n_ci.dot(n_w_in_c) - 1.;
//        A += gamma*(Ai * Ai.transpose());
//        b += gamma*(Ai * bi);
//        err += gamma*bi;
        // normal new
        Eigen::Matrix3f Asi = -T_wc.rotation().matrix()*tdp::SO3fda::invVee(n_ci);
        Eigen::Vector3f bsi = -(T_wc.rotation()*n_ci - n_w);
        A.topLeftCorner<3,3>() += gamma*(Asi*Asi.transpose());
        b.topRows<3>() += gamma*(Asi.transpose() * bsi);
        err += gamma*bsi.norm();
//        std::cout << Ai.transpose() << "; " << bi << std::endl;
        // texture
//        if (tdp::IsValidData(pl.grad_)) {
//          Eigen::Matrix<float,3,6> Jse3;
//          Jse3.leftCols<3>() = -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci));
//          Jse3.rightCols<3>() = Eigen::Matrix3f::Identity();
//          Ai = Jse3.transpose() * pl.grad_;
//          bi = grey_ci - pl.grey_;
//          A += lambda*(Ai * Ai.transpose());
//          b += lambda*(Ai * bi);
//          err += lambda*bi;
//        } else {
//          std::cout << "gradient is nan " << std::endl;
//        }
        // texture inverse transform verified Jse3 
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(T_cw*pc_w);
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << SO3mat<float>::invVee(T_cw.rotation()*(pc_w-T_wc.translation())), 
             -Eigen::Matrix3f::Identity();
        Ai = Jse3.transpose() * Jpi.transpose() * gradGrey_ci;
        bi = -grey_ci + pl.grey_;
        A += lambda*(Ai * Ai.transpose());
        b += lambda*(Ai * bi);
        err += lambda*bi;
        // accumulate
        return true;
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
    float p2plThr, 
    float dotThr,
    Eigen::Matrix<float,3,3>& N
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
  if (n_w_in_c.dot(n_ci) > dotThr) {
    float p2pl = n_w.dot(pc_w - T_wc*pc_ci);
    if (fabs(p2pl) < p2plThr) {
      N += n_w * n_ci.transpose();
      return true;
    }
  }
  return false;
}

template<int D>
bool CheckEntropyTermination(const Eigen::Matrix<float,D,D>& A,
    float Hprev,
    float HThr, float condEntropyThr, float negLogEvThr,
    float& H, bool verbose) {

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,D,D>> eig(A);
  Eigen::Matrix<float,D,1> negLogEv = -eig.eigenvalues().real().array().log();
  H = negLogEv.sum();
  if ((H < HThr || Hprev - H < condEntropyThr) 
      && (negLogEv.array() < negLogEvThr).all()) {
    if (verbose)
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
  std::string varsFile = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
    varsFile = (argc > 3) ? std::string(argv[3]) : "";
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
  pangolin::CreatePanel("visPanel").SetBounds(0.4,1.,pangolin::Attach::Pix(180),pangolin::Attach::Pix(360));
  pangolin::Display("visPanel").Show(false);
  pangolin::CreatePanel("mapPanel").SetBounds(0.4,1.,pangolin::Attach::Pix(180),pangolin::Attach::Pix(360));
  pangolin::Display("mapPanel").Show(false);

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
  size_t wc = w; //+w%64); // for convolution
  size_t hc = rig.NumCams()*h; //+h%64);
//  wc += wc%64;
//  hc += hc%64;

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

  pangolin::View& viewGrads = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  viewPc3D.AddDisplay(viewGrads);
  viewGrads.SetBounds(0.6,1.0,0.6,1.);

  tdp::QuickView viewCurrent(wc, hc);
//  gui.container().AddDisplay(viewCurrent);
  viewPc3D.AddDisplay(viewCurrent);
  viewCurrent.SetBounds(0.,0.3,0.,0.3);

  tdp::QuickView viewGreyGradNorm(wc, hc);
  viewPc3D.AddDisplay(viewGreyGradNorm);
  viewGreyGradNorm.SetBounds(0.,0.3,0.3,0.6);

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewGrey(3*wc/2, hc);
  containerTracking.AddDisplay(viewGrey);

  tdp::QuickView viewGradGrey(3*wc/2, hc);
  containerTracking.AddDisplay(viewGradGrey);

  tdp::QuickView viewD(3*wc/2, hc);
  containerTracking.AddDisplay(viewD);

  tdp::QuickView viewMask(3*wc/2, hc);
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
  viewGreyGradNorm.Show(true);
  plotters.Show(false);

  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<float> curv(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);
  tdp::ManagedHostImage<tdp::Vector4fda> dpc(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);

  tdp::ManagedHostImage<uint8_t> grey(w, h);
  tdp::ManagedHostPyramid<float,PYR> pyrGreyFlPrev(wc,hc);
  tdp::ManagedHostPyramid<float,PYR> pyrGreyFl(wc,hc);
  tdp::Image<float> greyFl = pyrGreyFl.GetImage(0);
  tdp::ManagedDeviceImage<uint8_t> cuGrey(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreyFl(wc,hc);
  tdp::ManagedHostImage<float> pyrGreyFlImg(3*wc/2, hc); 
  tdp::ManagedDevicePyramid<float,PYR> cuPyrGreyFlPrev(wc,hc);
  tdp::ManagedDevicePyramid<float,PYR> cuPyrGreyFlSmooth(wc,hc);
  tdp::Image<float> cuGreyFlSmooth = cuPyrGreyFlSmooth.GetImage(0);
  tdp::ManagedDeviceImage<float> cuGreyGradNorm(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyGradTheta(wc,hc);
  tdp::ManagedHostImage<float> greyGradNorm(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector2fda,PYR> cuPyrGradGrey(wc,hc);
  tdp::ManagedHostPyramid<tdp::Vector2fda,PYR> pyrGradGrey(wc,hc);
  tdp::Image<tdp::Vector2fda> cuGradGrey = cuPyrGradGrey.GetImage(0);
  tdp::Image<tdp::Vector2fda> gradGrey = pyrGradGrey.GetImage(0);

  tdp::ManagedDeviceImage<tdp::Vector2fda> cuGrad2D(3*wc/2, hc); 
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuGrad2DImg(3*wc/2, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> grad2DImg(3*wc/2, hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);

  tdp::ManagedHostPyramid<uint8_t,PYR> pyrMask(wc, hc);
  tdp::Image<uint8_t> mask = pyrMask.GetImage(0);
  tdp::ManagedHostPyramid<uint8_t,PYR> pyrMaskDisp(wc, hc);
  tdp::ManagedHostImage<uint8_t> pyrMaskImg(3*wc/2, hc);

  tdp::ManagedHostPyramid<uint32_t,PYR> pyrZ(w, h);
  tdp::Image<uint32_t> z = pyrZ.GetImage(0);

  tdp::ManagedHostImage<float> age(MAP_SIZE);

  tdp::ManagedHostPyramid<tdp::Vector3fda,PYR> pyrRay(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,PYR> cuPyrRay(wc,hc);
  tdp::ComputeCameraRays(cam, cuPyrRay);
  pyrRay.CopyFrom(cuPyrRay);

  // ICP stuff
  tdp::ManagedDevicePyramid<tdp::Vector3fda,PYR> cuPyrPc(wc,hc);
  tdp::Image<tdp::Vector3fda> cuPc = cuPyrPc.GetImage(0);
  tdp::ManagedHostPyramid<tdp::Vector3fda,PYR> pyrPc(wc,hc);
  tdp::Image<tdp::Vector3fda> pc = pyrPc.GetImage(0);
  pc.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  tdp::ManagedHostPyramid<tdp::Vector3fda,PYR> pyrN(wc,hc);
  tdp::Image<tdp::Vector3fda> n = pyrN.GetImage(0);

  tdp::ManagedDevicePyramid<float,PYR> cuPyrD(wc,hc);
  tdp::Image<float> cuD = cuPyrD.GetImage(0);
  tdp::ManagedHostPyramid<float,PYR> pyrD(wc,hc);
  tdp::Image<float> d = pyrD.GetImage(0);
  tdp::ManagedHostImage<float> pyrDImg(3*wc/2, hc); 

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

//  tdp::ManagedHostImage<tdp::Vector3fda> pc_c;
//  tdp::ManagedHostImage<tdp::Vector3bda> rgb_c;
//  tdp::ManagedHostImage<tdp::Vector3fda> n_c;

  pangolin::Var<bool> record("ui.record",false,true);
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,10.);
  pangolin::Var<bool> showVisPanel("ui.viz panel",false,true);
  pangolin::Var<bool> showMapPanel("ui.map panel",false,true);

  pangolin::Var<bool> savePly("ui.save ply",false,true);

  pangolin::Var<int> numMapPoints("ui.num Map",0,0,0);
  pangolin::Var<int> numProjected("ui.num Proj",0,0,0);
  pangolin::Var<int> numInl("ui.num Inl",0,0,0);
  pangolin::Var<int> idMapUpdate("ui.id Map",0,0,0);
  pangolin::Var<int> idNNUpdate("ui.id NN",0,0,0);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> runLoopClosureGeom("ui.run loop closure geom",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updateMap("ui.update map",true,true);
  // TODO if sample normals if off then doRegvMF shoudl be on
  pangolin::Var<bool> sampleNormals("ui.sampleNormals",true,true);

  pangolin::Var<bool> pruneNoise("ui.prune Noise",false,true);
  pangolin::Var<int> survivalTime("ui.survival Time",100,0,200);
  pangolin::Var<int> minNumObs("ui.min Obs",10,1,20);
  pangolin::Var<int> numAdditionalObs("ui.num add Obs",300,0,1000);

  pangolin::Var<int> smoothGrey("ui.smooth grey",1,0,2);
  pangolin::Var<int> smoothGreyPyr("ui.smooth grey pyr",1,0,1);
  pangolin::Var<int> smoothDPyr("ui.smooth D pyr",1,0,1);
  pangolin::Var<bool> normalViaRMLS("ui.normal RMLS",false,true);
  pangolin::Var<int>  W("ui.W ",9,1,15);
  pangolin::Var<float> subsample("ui.subsample %",1.,0.1,3.);
  pangolin::Var<float> pUniform("ui.p uniform ",0.1,0.1,1.);

  pangolin::Var<bool> doRegvMF("mapPanel.reg vMF",false,true);
  pangolin::Var<bool> doRegPc0("mapPanel.reg pc0",true,true);
  pangolin::Var<bool> doRegAbsPc("mapPanel.reg abs pc",true,true);
  pangolin::Var<bool> doRegAbsN("mapPanel.reg abs n",false,true);
  pangolin::Var<bool> doRegRelPlZ("mapPanel.reg rel Pl",true,true);
  pangolin::Var<bool> doRegRelNZ("mapPanel.reg rel N",true,true);
//  pangolin::Var<bool> doRegRelPlObs("mapPanel.reg rel PlObs",false,true);
//  pangolin::Var<bool> doRegRelNObs("mapPanel.reg rel NObs",false,true);
  pangolin::Var<bool> doVariationalUpdate("mapPanel.variational",true,true);
  pangolin::Var<bool> useMrfInVariational("mapPanel. use MRF in var",false,true);
  pangolin::Var<float> lambdaRegDir("mapPanel.lamb Reg Dir",0.01,0.01,1.);
  pangolin::Var<float> lambdaMRF("mapPanel.lamb z MRF",.1,0.01,10.);
  pangolin::Var<float> alphaGrad("mapPanel.alpha Grad",.0001,0.0,1.);
  pangolin::Var<float> tauO("mapPanel.tauO",100.,0.0,200.);
  pangolin::Var<float> tauP("mapPanel.tauP",100.,0.0,200.);
  pangolin::Var<float> sigmaPl("mapPanel.sigmaPl",0.03,0.01,.2);
  pangolin::Var<float> sigmaPc0("mapPanel.sigmaPc0",0.03,0.01,0.1);
  pangolin::Var<float> sigmaObsP("mapPanel.sigmaObsP",0.06,0.01,0.2);
  pangolin::Var<float> maxNnDist("mapPanel.max NN Dist",0.2, 0.1, 1.);

  pangolin::Var<bool> runICP("ui.run ICP",true,true);
  pangolin::Var<bool> icpReset("ui.reset icp",true,false);
  pangolin::Var<int> maxIt0("ui.max iter 0",10, 1, 20);
  pangolin::Var<int> maxIt1("ui.max iter 1",7, 1, 20);
  pangolin::Var<int> maxIt2("ui.max iter 2",5, 1, 20);
  pangolin::Var<int> maxIt3("ui.max iter 3",5, 1, 20);
  pangolin::Var<int> ICPmaxLvl("ui.icp max lvl",1, 0, PYR-1);

  pangolin::Var<bool> pruneAssocByRender("ui.prune assoc by render",true,true);
  pangolin::Var<bool> semanticObsSelect("ui.semObsSelect",true,true);
  pangolin::Var<bool> sortByGradient("ui.sortByGradient",true,true);

  pangolin::Var<int> dtAssoc("ui.dtAssoc",5000,1,1000);
  pangolin::Var<float> lambdaNs("ui.lamb Ns",0.1,0.001,1.);
  pangolin::Var<float> lambdaTex("ui.lamb Tex",0.1,0.01,1.);
  pangolin::Var<bool> useTexture("ui.use Tex ICP",true,true);
  pangolin::Var<bool> use3dGrads("ui.use 3D grads ",false,true);
  pangolin::Var<bool> useNormals("ui.use Ns ICP",false,true);
  pangolin::Var<bool> useNormalsAndTexture("ui.use Tex&Ns ICP",false,true);
  pangolin::Var<bool> usevMFmeans("ui.use vMF means",false,true);

  pangolin::Var<float> occlusionDepthThr("ui.occlusion D Thr",0.06,0.01,0.3);
  pangolin::Var<float> angleThr("ui.angle Thr",15, -1, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.03,0,0.3);
  pangolin::Var<float> HThr("ui.H Thr",-32.,-40.,-12.);
  pangolin::Var<float> negLogEvThr("ui.neg log ev Thr",-4.,-12.,-1.);
  pangolin::Var<float> dPyrHThr("ui.d Pyr H Thr",4.,0.,8.);
  pangolin::Var<float> dPyrNewLogEvHThr("ui.d Pyr H Thr",1.,0.,3.);
  pangolin::Var<float> dPyrdAlpha("ui.d Pyr dAlpha",0.9,0.1,1.);
  pangolin::Var<float> condEntropyThr("ui.rel log dH ", 1.e-3,1.e-3,1e-2);
  pangolin::Var<float> icpdRThr("ui.dR Thr",0.25,0.1,1.);
  pangolin::Var<float> icpdtThr("ui.dt Thr",0.01,0.01,0.001);

  pangolin::Var<bool> doSO3prealign("ui.SO3 prealign",true,true);
  pangolin::Var<bool> useGpuPrealign("ui.GPU prealign",true,true);
  pangolin::Var<float> SO3HThr("ui.SO3 H Thr",-24.,-40.,-20.);
  pangolin::Var<float> SO3negLogEvThr("ui.SO3 neg log ev Thr",-6.,-10.,0.);
  pangolin::Var<float> SO3condEntropyThr("ui.SO3 rel log dH ", 1.e-3,1.e-6,1e-2);
  pangolin::Var<int> SO3maxIt("ui.SO3 max iter",2, 1, 20);
  pangolin::Var<int> SO3maxLvl("ui.SO3 max Lvl",PYR-1,0,PYR-1);
  pangolin::Var<int> SO3minLvl("ui.SO3 min Lvl",1,0,PYR-1);

  pangolin::Var<float> scale("visPanel.scale",0.05,0.1,1);
  pangolin::Var<int> step("visPanel.step",10,1,100);
  pangolin::Var<float> bgGrey("visPanel.bg Grey",0.02,0.0,1);
  pangolin::Var<bool> showGradDir("visPanel.showGradDir",true,true);
  pangolin::Var<bool> showPlanes("visPanel.show planes",false,true);
  pangolin::Var<bool> showPcModel("visPanel.show model",false,true);
  pangolin::Var<bool> showPcCurrent("visPanel.show current",false,true);
  pangolin::Var<int> showPcLvl("visPanel.cur Lvl",0,0,PYR-1);
  pangolin::Var<bool> showFullPc("visPanel.show full",true,true);
  pangolin::Var<bool> showNormals("visPanel.show ns",true,true);
  pangolin::Var<bool> showGrads("visPanel.show grads",false,true);
  pangolin::Var<bool> showPcEst("visPanel.show PcEst",false,true);
  pangolin::Var<bool> showAge("visPanel.show age",false,true);
  pangolin::Var<bool> showObs("visPanel.show # obs",false,true);
  pangolin::Var<bool> showCurv("visPanel.show curvature",false,true);
  pangolin::Var<bool> showGrey("visPanel.show grey",false,true);
  pangolin::Var<bool> showNumSum("visPanel.show numSum",false,true);
  pangolin::Var<bool> showZCounts("visPanel.show zCounts",false,true);
  pangolin::Var<bool> showHp("visPanel.show Hp",false,true);
  pangolin::Var<bool> showHn("visPanel.show Hn",false,true);
  pangolin::Var<bool> showInfoObs("visPanel.show InfoObs",false,true);
  pangolin::Var<bool> showLabels("visPanel.show labels",true,true);
  pangolin::Var<bool> showSamples("visPanel.show Samples",false,true);
  pangolin::Var<bool> showSurfels("visPanel.show surfels",true,true);
  pangolin::Var<bool> showNN("visPanel.show NN",false,true);
  pangolin::Var<bool> showLoopClose("visPanel.show loopClose",false,true);

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
  pangolin::GlBuffer gradbo_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,3);
  pangolin::GlBuffer rbo(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,1);
  pangolin::GlBuffer tbo(pangolin::GlArrayBuffer,MAP_SIZE,GL_UNSIGNED_SHORT,1);
  pangolin::GlBuffer lbo(pangolin::GlArrayBuffer,MAP_SIZE,GL_UNSIGNED_SHORT,1);
  pangolin::GlBuffer cbo_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_UNSIGNED_BYTE,3);
  pangolin::GlBuffer valuebo(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,1);
  pangolin::GlBuffer vboEst_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_FLOAT,3);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> rs(MAP_SIZE); // radius of surfels
  tdp::ManagedHostCircularBuffer<uint16_t> ts(MAP_SIZE); // radius of surfels
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> grad_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> gradDir_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pcEst_w(MAP_SIZE);

  rs.Fill(NAN);
  ts.Fill(0);
  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  rgb_w.Fill(tdp::Vector3bda::Zero());
  grad_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  gradDir_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  n_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  
  vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
  nbo_w.Upload(n_w.ptr_, n_w.SizeBytes(), 0);
  cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeBytes(), 0);
  gradbo_w.Upload(grad_w.ptr_, grad_w.SizeBytes(), 0);
  tbo.Upload(ts.ptr_, ts.SizeBytes(), 0);
  vboEst_w.Upload(pcEst_w.ptr_, pcEst_w.SizeBytes(), 0);

  tdp::ManagedHostCircularBuffer<uint8_t> nnFixed(MAP_SIZE);
  nnFixed.Fill(0);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNida> nn(MAP_SIZE);
  nn.Fill(tdp::VectorkNNida::Ones()*-1);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> sumSameZ(MAP_SIZE);
  sumSameZ.Fill(tdp::VectorkNNfda::Zero());
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> numSamplesZ(MAP_SIZE);
  numSamplesZ.Fill(tdp::VectorkNNfda::Zero());
//  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> mapObsNum(MAP_SIZE);
//  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> mapObsDot(MAP_SIZE);
//  tdp::ManagedHostCircularBuffer<tdp::VectorkNNfda> mapObsP2Pl(MAP_SIZE);
//  mapObsNum.Fill(tdp::VectorkNNfda::Zero());

  tdp::ManagedHostCircularBuffer<tdp::Matrix3fda> outerSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pcSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> numSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> normSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> infoObsSum(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc0_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Matrix3fda> pc0Info_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Matrix3fda> pcObsInfo_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pcObsXi_w(MAP_SIZE);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> Jn_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> Jp_w(MAP_SIZE);

  tdp::ManagedHostImage<uint16_t> labels(MAP_SIZE);

  std::vector<std::pair<int32_t, int32_t>> mapNN;
  mapNN.reserve(MAP_SIZE*kNN);

  int32_t iReadCurW = 0;
  int32_t frame = 0;

  tdp::ThreadedValue<bool> runSampling(true);
  tdp::ThreadedValue<bool> runTopologyThread(true);
  tdp::ThreadedValue<bool> runMappingThread(true);

  std::mutex pl_wLock;
  std::mutex nnLock;
  std::mutex mapLock;
  std::thread topology([&]() {
    int32_t iReadNext = 0;
    int32_t sizeToReadPrev = 0;
    int32_t sizeToRead = 0;
    std::deque<int32_t> newIds;
    tdp::VectorkNNfda values;
    std::mt19937 rnd(0);
    while(runTopologyThread.Get()) {
      sizeToReadPrev = sizeToRead;
      {
        std::lock_guard<std::mutex> lock(pl_wLock); 
        sizeToRead = pl_w.SizeToRead();
      }
      if (sizeToRead ==0) continue;
      if (sizeToRead == sizeToReadPrev) {
        if (newIds.size() > 0) {
          iReadNext = newIds.front();
          newIds.pop_front();
        } else {
          std::uniform_int_distribution<int32_t> unif(0, sizeToRead-1);
          iReadNext = unif(rnd);
        }
      } else {
        for (int32_t i=sizeToReadPrev+1; i<sizeToRead; ++i)
          newIds.push_back(i);
        iReadNext = sizeToReadPrev;
      }
      if (nnFixed[iReadNext] < kNN) {
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        if (pruneNoise && pl.lastFrame_+survivalTime < frame && pl.numObs_ < minNumObs) {
          pc_w[iReadNext] = tdp::Vector3fda(NAN,NAN,NAN);
          n_w[iReadNext]  = tdp::Vector3fda(NAN,NAN,NAN);
          pl.valid_ = false;
        }
        values.fill(std::numeric_limits<float>::max());
        tdp::VectorkNNida& ids = nn[iReadNext];
        tdp::VectorkNNida idsPrev = ids;
        ids = tdp::VectorkNNida::Ones()*(-1);
        for (int32_t i=0; i<sizeToRead; ++i) {
          if (i != iReadNext) {
            float dist = (pl.p_-pl_w.GetCircular(i).p_).squaredNorm();
            tdp::AddToSortedIndexList<kNN>(ids, values, i, dist);
//            std::cout << i << ", " << dist << "| " <<  ids.transpose() << " : " << values.transpose() << std::endl;
          }
        }
        // for map constraints
        // TODO: should be updated as pairs are reobserved
        nnFixed[iReadNext] = kNN;
        for (int32_t i=0; i<kNN; ++i) {
//            mapObsDot[iReadNext][i] = pl.n_.dot(pl_w[ids[i]].n_);
//            mapObsP2Pl[iReadNext][i] = pl.p2plDist(pl_w[ids[i]].p_);
//            mapObsNum[iReadNext][i] = 1;
            if (ids(i) != idsPrev(i)) {
              numSamplesZ[iReadNext][i] = 0;
              sumSameZ[iReadNext][i] = 0;
//              mapObsDot[iReadNext][i] = 0.;
//              mapObsP2Pl[iReadNext][i] = 0.;
//              mapObsNum[iReadNext][i] = 0.;
//  //            std::cout << "resetting " << iReadNext << " " << i << std::endl;
            }
          if (values(i) > maxNnDist*maxNnDist) {
            ids(i) = -1;
            nnFixed[iReadNext]-- ;
          }
        }
        // just for visualization
        if (mapNN.size() < kNN*iReadNext) {
          for (int32_t i=0; i<kNN; ++i) 
            mapNN.emplace_back(iReadNext, ids(i));
        } else {
          for (int32_t i=0; i<kNN; ++i) 
            mapNN[iReadNext*kNN+i] = std::pair<int32_t,int32_t>(iReadNext, ids[i]);
        }
      }
      iReadNext = (iReadNext+1)%sizeToRead;
      {
        std::lock_guard<std::mutex> lock(nnLock); 
//          mapObsDot.iInsert_ = std::max(iReadNext;
//          mapObsP2Pl.iInsert_ = std::max(iReadNext;
        nn.iInsert_ = std::max(iReadNext, nn.iInsert_);
      }
      idNNUpdate = iReadNext;
    };
  });


  uint32_t K = 0;
  std::mutex zsLock;
  std::mutex vmfsLock;
  std::mt19937 rnd(910481);
  float logAlpha = log(.1);
//  Eigen::Matrix3f SigmaO = 0.0001*Eigen::Matrix3f::Identity();
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
    int32_t iInsert = 0;
//    std::random_device rd_;
    std::mt19937 rnd(0);
    while(runSampling.Get()) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        iInsert = nn.iInsert_;
      }
      if (iInsert == 0) continue;
      pS.iInsert_ = nn.iInsert_;
      nS.iInsert_ = nn.iInsert_;
      // sample normals using dpvmf and observations from planes
      size_t Ksample = vmfs.size();
      vmfSS.Fill(tdp::Vector4fda::Zero());
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        uint16_t& zi = zS[i];
        tdp::Vector3fda& ni = nS[i];
        if (!pl_w[i].valid_)
          continue;
        if (sampleNormals) {
          //tdp::plane& pl = pl_w[i];
          //eigen::vector3f mu = pl.w_*pl.n_*tauo;
          //std::cout << pl.w_ * pl.n_.transpose() << " " 
          //  << nsum_w[i].transpose() << std::endl;
          tdp::Vector3fda mu = normSum_w[i]*nSum_w[i]*tauO;
          if (zi < Ksample) {
            mu += vmfs[zi].mu_*vmfs[zi].tau_;
          }
          ni = vMF<float,3>(mu).sample(rnd);
          pl_w[i].n_ = ni;
        } else {
          ni = pl_w[i].n_;
        }
        vmfSS[zi].topRows<3>() += ni;
        vmfSS[zi](3) ++;
      }
      // sample dpvmf labels
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        if (!pl_w[i].valid_)
          continue;
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
      std::vector<uint32_t> labelMap(Ksample);
      std::iota(labelMap.begin(), labelMap.end(), 0);
      size_t j=0;
      for (size_t k=0; k<Ksample; ++k) {
        labelMap[k] = j;
        if (vmfSS[k](3) > 0) {
          j++;
        }
      }
      // sample dpvmf parameters
      {
        std::lock_guard<std::mutex> lock(vmfsLock);
        std::lock_guard<std::mutex> lockZs(zsLock);
        for (size_t k=0; k<Ksample; ++k) {
          if (vmfSS[k](3) > 0) {
            vmfs[labelMap[k]] = base.posterior(vmfSS[k]).sample(rnd);
            vmfSS[labelMap[k]] = vmfSS[k];
          }
        }
        Ksample = j;
        vmfs.resize(Ksample);
//      }
//      {
        for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
          zS[i] = labelMap[zS[i]];
          pl_w[i].z_ = zS[i];
        }
        K = Ksample;
      }

      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        uint16_t& zi = zS[i];
        for (size_t k=0; k<kNN; ++k) {
          if (0 <= nn[i](k) && nn[i](k) < iInsert) {
            numSamplesZ[i](k) ++;
            if(zS[i] == zS[nn[i](k)])
              sumSameZ[i]++
          }
        }
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

  std::thread mapping([&]() {
    int32_t i = 0;
    int32_t sizeToRead = 0;
//    std::random_device rd_;
    std::mt19937 gen_(0);
    while(runMappingThread.Get()) {
    if (updateMap) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        sizeToRead = nn.SizeToRead();
      }
      if (sizeToRead == 0) continue;
      tdp::Plane& pl = pl_w.GetCircular(i);
      if (!pl.valid_) continue;
      // compute gradient
      tdp::VectorkNNida& ids = nn.GetCircular(i);
//      tdp::Vector3fda& Jn = Jn_w[i];
//      tdp::Vector3fda& Jp = Jp_w[i];
      tdp::Vector3fda Jn = tdp::Vector3fda::Zero();
      tdp::Vector3fda Jp = tdp::Vector3fda::Zero();
      tdp::Vector3fda mu = tdp::Vector3fda::Zero();
      float tau = 0.;
      vmfsLock.lock();
      if (pl.z_ < K && vmfs[pl.z_].tau_ > 0) {
        mu = vmfs[pl.z_].mu_;
        tau= vmfs[pl.z_].tau_;
      }
      vmfsLock.unlock();
      if (doRegvMF && lambdaRegDir > 0) {
        Jn = -lambdaRegDir*mu*tau;
        if (!tdp::IsValidData(Jn)) {
//          std::cout << Jn.transpose() << " " << pl.z_ <<
//            vmfs[pl.z_].mu_.transpose() << ", " << vmfs[pl.z_].tau_ << std::endl;
          Jn = tdp::Vector3fda::Zero();
        }
      }
      if (doRegPc0) {
        Jp += -1./(sigmaPc0*sigmaPc0)*(pc0_w[i] - pl.p_);
      }
      if (doRegAbsPc) {
        Jp += pl.n_ *pl.n_.dot(pl.p_*infoObsSum[i] - pcSum_w[i]);
//        Jp += pl.n_ *numSum_w[i]*pl.n_.dot(pl.p_ - pcSum_w[i]);
//        Jp += 2.*(numSum_w[i]*pl.p_ - numSum_w[i]*pcSum_w[i]);
      }
//      if (doRegAbsN) {
////        Jn += (numSum_w[i]*pl.n_ - normSum_w[i]*nSum_w[i]);
//        Jn += -tauO*normSum_w[i]*nSum_w[i];
//        Jn += numSum_w[i]*(outerSum_w[i]*pl.n_ - pl.n_.dot(pcSum_w[i])*pl.p_ - pl.n_.dot(pl.p_)*(pcSum_w[i] - pl.p_));
//      }
      // TODO this seems to not work!
      bool haveFullNeighborhood = (ids.array() >= 0).all();
      if (haveFullNeighborhood && !doVariationalUpdate) {
        tdp::Vector3fda JpMRF = tdp::Vector3fda::Zero();
        float wMRF = 0.f;
        for (int j=0; j<kNN; ++j) {
          if (ids[j] > -1 && pl_w[ids[j]].valid_) {
            const tdp::Plane& plO = pl_w[ids[j]];
            if (doRegRelPlZ) {
              if (pl.z_ == plO.z_) {
                float wi = exp(-(pl.p_-plO.p_).squaredNorm());
                wMRF += wi;
                JpMRF += -wi/(sigmaPl*sigmaPl)*(pl.p2plDist(plO.p_))*pl.n_;
//                JpMRF += -1./(sigmaPl*sigmaPl)*(pl.p2plDist(plO.p_))*pl.n_;
                Jn +=  1./(sigmaPl*sigmaPl)*(pl.p2plDist(plO.p_))*(plO.p_-pl.p_);
              }
            }
            if (doRegRelNZ) {
              if (pl.z_ == plO.z_) {
//                Jn +=  2.*lambdaRegPl*tau*(pl.n_ - plO.n_);
                Jn += -tauP*plO.n_;
              }
            }
//            if (mapObsNum[i](j) > 0.) {
//              if (doRegRelNObs) {
//                Jn += mapObsNum[i](j)*2.*(pl.n_.dot(plO.n_)-mapObsDot[i](j)/mapObsNum[i](j))*plO.n_;
//              }
//              if (doRegRelPlObs) {
//                Jn += mapObsNum[i](j)*2.*(pl.p2plDist(plO.p_)-mapObsP2Pl[i](j)/mapObsNum[i](j))*(plO.p_-pl.p_);
//                Jp += -mapObsNum[i](j)*2.*(pl.p2plDist(plO.p_)-mapObsP2Pl[i](j)/mapObsNum[i](j))*pl.n_;
//              }
//            }
          }
        }
        if (doRegRelPlZ && wMRF > 0.) {
//          std::cout << wMRF << " " << JpMRF.transpose() << std::endl;
          Jp += JpMRF/wMRF;
//          Jp += JpMRF;
        }
      }
      tdp::Vector3fda pmu;
        if (doVariationalUpdate && haveFullNeighborhood) {
          Eigen::Matrix3f InfoPl;
          tdp::Matrix3fda& InfoO = pc0Info_w[i]; //1./(sigmaPc0*sigmaPc0)*Eigen::Matrix3f::Identity();
          Eigen::Matrix3f Info = InfoO + numSum_w[i]* pcObsInfo_w[i]; // + numSum_w[i]*pl_w[i].n_*pl_w[i].n_.transpose()*infoObsSum[i];
          Eigen::Vector3f xi = InfoO*pc0_w[i] + numSum_w[i]*pcObsXi_w[i]; // + numSum_w[i]*pl_w[i].n_*pl_w[i].n_.transpose()*pcSum_w[i]; //*pl.w_;
          if (useMrfInVariational) {
            for (int i=0; i<kNN; ++i) {
              if (ids[i] > -1 && pl_w[ids[i]].valid_) {
                InfoPl = tau/(sigmaPl*sigmaPl)*pl_w[ids[i]].n_*pl_w[ids[i]].n_.transpose();
                //InfoPl = 1./(sigmaPl*sigmaPl)*pl_w[ids[i]].n_*pl_w[ids[i]].n_.transpose();
                //InfoPl = pl_w[i].n_*pl_w[i].n_.transpose();
                Info += InfoPl;
                xi += InfoPl*pl_w[ids[i]].p_;
              } else {
                std::cout << "have full neighborhood but id " <<  ids[i] << std::endl;
              }
            }
          }
          pmu = Info.ldlt().solve(xi);
          pl.Info_ = Info;
          pl.Hp_ = -Info.eigenvalues().real().array().log().sum();
          if ( (pmu-pl.p_).norm() > 0.1) {
            std::cout << pl.Hp_  << " " << numSum_w[i]
              << ": " << pl_w[i].p_.transpose() << " " << pmu.transpose() << " xi " << xi.transpose()  << std::endl;
            std::cout << Info.eigenvalues().real().transpose() << std::endl;
            std::cout << InfoO << std::endl;
            std::cout << Info << std::endl;
          }
        }
      // apply gradient
//      tdp::Vector3fda& Jn = Jn_w[i];
//      tdp::Vector3fda& Jp = Jp_w[i];
      {
        std::lock_guard<std::mutex> mapGuard(mapLock);
        if (!sampleNormals) 
          pl.n_ = (pl.n_- alphaGrad * Jn).normalized();
        if (haveFullNeighborhood &&  doVariationalUpdate) {
          pl.p_ = pmu;
        } else {
          pl.p_ -= alphaGrad * Jp;
        }
        pc_w[i] = pl.p_;
        n_w[i] = pl.n_;
      }
      i = (i+1)%sizeToRead;
      idMapUpdate = i;
//      std::cout << "map updated " << i << " " 
//        << (alphaGrad * Jn.transpose()) << "; "
//        << (alphaGrad * Jp.transpose()) << std::endl;
    }
    };
  });

  tdp::ProjectiveAssociation<CameraT::NumParams, CameraT> projAssoc(cam, w, h);

  std::vector<std::pair<size_t, size_t>> assoc;
  assoc.reserve(10000);

  std::vector<std::vector<std::vector<uint32_t>>*> invInd;
  for (size_t lvl=0; lvl<PYR; ++lvl) {
    invInd.push_back(new std::vector<std::vector<uint32_t>>());
  }
  std::vector<size_t> id_w;
  id_w.reserve(MAP_SIZE);

//  std::random_device rd;
  std::mt19937 gen(19023);

  std::vector<uint32_t> idNew;
  idNew.reserve(w*h);

  pyrMask.Fill(0);
  std::vector<std::vector<uint32_t>*> idsCur;
  for (size_t lvl=0; lvl<PYR; ++lvl) {
    idsCur.push_back(new std::vector<uint32_t>());
    idsCur.back()->reserve(w*h);
  }

  std::ofstream out("trajectory_tumFormat.csv");
  out << "# " << input_uri << std::endl;

  if (varsFile.size() > 0)
    pangolin::LoadJsonFile(varsFile, "");

  pangolin::SaveJsonFile("./varsUi.json", "ui");
  pangolin::SaveJsonFile("./varsMap.json", "mapPanel");
  pangolin::SaveJsonFile("./varsVis.json", "visPanel");

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (showVisPanel.GuiChanged()) {
      pangolin::Display("visPanel").Show(showVisPanel);
    }
    if (showMapPanel.GuiChanged()) {
      pangolin::Display("mapPanel").Show(showMapPanel);
    }

    if (runLoopClosureGeom.GuiChanged()) {
      showLoopClose = runLoopClosureGeom;
    }
    if (pangolin::Pushed(icpReset)) {
      T_wc = tdp::SE3f();
    }
    if (pangolin::Pushed(savePly)) {
      std::string path = pangolin::MakeUniqueFilename("./surfelMap.ply");
      std::vector<std::string> comments;
      comments.push_back(input_uri);
      comments.push_back(calibPath);
      tdp::SavePointCloud(path, pc_w, n_w, rgb_w, true, comments);
    }

    idNew.clear();
    if (!gui.paused() && !gui.finished()
        && frame > 0
        && runMapping
        && (trackingGood || frame == 1)) { // add new observations
      std::cout << " adding new planes to map " << std::endl;

      // update mask only once to know where to insert new planes
      TICK("data assoc");
      projAssoc.Associate(vbo_w, nbo_w, tbo, T_wc.Inverse(), dMin,
          dMax, std::max(0, frame-dtAssoc), pl_w.SizeToRead());
      TOCK("data assoc");
      TICK("extract assoc");
//      z.Fill(0);
      for (size_t lvl=0; lvl<PYR; ++lvl) idsCur[lvl]->clear();
//      projAssoc.GetAssoc(z, mask, idsCur);
//      projAssoc.GetAssocOcclusion(pl_w, pc, T_wc.Inverse(),
//          occlusionDepthThr, z, mask, idsCur);
      projAssoc.GetAssocOcclusion(pl_w, pyrPc, T_wc.Inverse(),
          occlusionDepthThr, dMin, dMax, pyrZ, pyrMask, idsCur);
      TOCK("extract assoc");
      pyrMaskDisp.CopyFrom(pyrMask);

      TICK("mask");
      tdp::GradientNormBiasedResampleEmptyPartsOfMask(pc, cam, mask,
          greyGradNorm, W, subsample, gen, 32, 32, w, h, pUniform, idNew);
//      tdp::UniformResampleEmptyPartsOfMask(pc, cam, mask, W,
//          subsample, gen, 32, 32, w, h);
      TOCK("mask");
      {
        iReadCurW = pl_w.iInsert_;
        std::lock_guard<std::mutex> lock(pl_wLock); 
        TICK("normals");
        ExtractPlanes(pc, rgb, grey, greyFl, gradGrey,
             mask, W, frame, T_wc, cam, dpc, pl_w, pc_w, pc0_w, pc0Info_w, rgb_w,
            n_w, grad_w, rs, ts, normalViaRMLS);

        std::cout << " extracted " << pl_w.iInsert_-iReadCurW << " new planes " << std::endl;
        TOCK("normals");
        TICK("add to model");
        for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
          gradDir_w[i] = pl_w[i].grad_.normalized();
          infoObsSum[i] = pl_w[i].n_.dot(pc0Info_w[i]*pl_w[i].n_) ; // 1./(sigmaObsP*sigmaObsP);
          pcSum_w[i] = pl_w[i].p_*infoObsSum[i];
//          pcSum_w[i] = pl_w[i].p_/(sigmaObsP*sigmaObsP);
         
          pcObsInfo_w[i] = pc0Info_w[i];
          pcObsXi_w[i] = pc0Info_w[i]* pl_w[i].p_;

          outerSum_w[i] = pl_w[i].p_*pl_w[i].p_.transpose();
          nSum_w[i] = pl_w[i].n_;
          numSum_w[i] = 1;
          normSum_w[i] = 1;
          pcEst_w[i] = pl_w[i].p_;
          idsCur[0]->emplace_back(i);
        }
      }
//      vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
      vbo_w.Upload(&pc_w.ptr_[iReadCurW], 
          pc_w.SizeToRead(iReadCurW)*sizeof(tdp::Vector3fda), 
          iReadCurW*sizeof(tdp::Vector3fda));
      nbo_w.Upload(&n_w.ptr_[iReadCurW], 
          pc_w.SizeToRead(iReadCurW)*sizeof(tdp::Vector3fda), 
          iReadCurW*sizeof(tdp::Vector3fda));
      rbo.Upload(&rs.ptr_[iReadCurW],
          rs.SizeToRead(iReadCurW)*sizeof(float),
          iReadCurW*sizeof(float));
      tbo.Upload(&ts.ptr_[iReadCurW],
          ts.SizeToRead(iReadCurW)*sizeof(uint16_t),
          iReadCurW*sizeof(int16_t));

      for (size_t lvl=0; lvl<PYR; ++lvl) {
        std::random_shuffle(idsCur[lvl]->begin(), idsCur[lvl]->end());
        std::cout << "@" << lvl <<  " lvl projected " 
          << idsCur[lvl]->size() 
          << " of " << pl_w.SizeToRead() << std::endl;
      }

      TOCK("add to model");
      numMapPoints = pl_w.SizeToRead();
      TICK("inverseIndex");
      {
        std::lock_guard<std::mutex> lockZs(zsLock);
        for (size_t lvl=0; lvl<PYR; ++lvl) {
          for (size_t k=0; k<K+1; ++k) {
            if (k >= invInd[lvl]->size()) {
              invInd[lvl]->push_back(std::vector<uint32_t>());
              invInd[lvl]->back().reserve(3000);
            } else {
              invInd[lvl]->at(k).clear();
            }
          }
        }
        if (semanticObsSelect && frame > 10) {
          if (pruneAssocByRender) {
            // only use ids that were found by projecting into the current pose
            for (size_t lvl=0; lvl<PYR; ++lvl) {
              for (auto i : *idsCur[lvl]) {
                uint32_t k = pl_w[i].z_;
                if (invInd[lvl]->at(k).size() < 3000)
                  invInd[lvl]->at(k).push_back(i);
              }
            }
          } else {      
            id_w.resize(pl_w.SizeToRead());
            std::iota(id_w.begin(), id_w.end(), 0);
            std::random_shuffle(id_w.begin(), id_w.end());
            // use all ids in the current map
            for (size_t lvl=0; lvl<PYR; ++lvl) {
              for (auto i : id_w) {
                uint32_t k = pl_w[i].z_;
                if (invInd[lvl]->at(k).size() < 3000)
                  invInd[lvl]->at(k).push_back(i);
              }
            }
          }
        } else {
          uint32_t k = 0;
          if (pruneAssocByRender) {
            // only use ids that were found by projecting into the current pose
            for (size_t lvl=0; lvl<PYR; ++lvl) {
              for (auto i : *idsCur[lvl]) {
                if (invInd[lvl]->at(k).size() < 3000)
                  invInd[lvl]->at(k).push_back(i);
                k = (k+1)%invInd[lvl]->size();
              }
            }
          } else {      
            id_w.resize(pl_w.SizeToRead());
            std::iota(id_w.begin(), id_w.end(), 0);
            std::random_shuffle(id_w.begin(), id_w.end());
            // use all ids in the current map
            for (size_t lvl=0; lvl<PYR; ++lvl) {
              for (auto i : id_w) {
                if (invInd[lvl]->at(k).size() < 3000)
                  invInd[lvl]->at(k).push_back(i);
                k = (k+1)%invInd[lvl]->size();
              }
            }
          }
        }
        if (sortByGradient) {
          // TODO realy this should look at the current image? although
          // maybe not since that is not even propperly aligned yet 
            for (size_t lvl=0; lvl<PYR; ++lvl)
              for (size_t k=0; k<invInd[lvl]->size(); ++k) {
                std::sort(invInd[lvl]->at(k).begin(), invInd[lvl]->at(k).end(),
                    [&](uint32_t ida, uint32_t idb) {
                    return pl_w[ida].gradNorm_ > pl_w[idb].gradNorm_;
                    });
          }
        }
      }
      std::cout << " inverse index " << invInd.size()  << std::endl;
      for (size_t lvl=0; lvl<PYR; ++lvl) {
        std::cout << "@" << lvl << " lvl: " << idsCur[lvl]->size() << " " << id_w.size() << std::endl;
        std::cout << "      ";
        for (size_t k=0; k<invInd[lvl]->size(); ++k) 
          std::cout << invInd[lvl]->at(k).size() << " ";
        std::cout << std::endl;
      }
      std::cout << " inverse done" << std::endl;
      TOCK("inverseIndex");
    }

//    glClearColor(bgGrey, bgGrey, bgGrey, 1.0f);
//    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    gui.NextFrames();

    int64_t t_host_us_d = 0;
    TICK("Setup");
    if (gui.verbose) std::cout << "collect d" << std::endl;
    cuD = cuPyrD.GetImage(0);
    rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
    if (smoothDPyr==1) {
      tdp::CompletePyramidBlur9(cuPyrD, 1.);
    } else {
      tdp::CompletePyramidBlur(cuPyrD, 1.);
    }
    pyrD.CopyFrom(cuPyrD);
    d = pyrD.GetImage(0);
    if (gui.verbose) std::cout << "compute pc" << std::endl;

    tdp::SE3f T_rc;
    for (size_t pyr=0; pyr<PYR; ++pyr) {
      tdp::Image<float> cuDLvl = cuPyrD.GetImage(pyr);
      tdp::Image<tdp::Vector3fda> cuPcLvl = cuPyrPc.GetImage(pyr);
      CameraT camLvl = cam.Scale(pow(0.5,pyr));
      tdp::Depth2PCGpu(cuDLvl, camLvl, T_rc, cuPcLvl);
    }
    cuPc = cuPyrPc.GetImage(0);
//    rig.ComputePc(cuD, true, cuPc);
//    tdp::CompletePyramid(cuPyrPc);
//    pc.CopyFrom(cuPyrPc.GetImage(0));
    pyrPc.CopyFrom(cuPyrPc);
    pc = pyrPc.GetImage(0);
    if (gui.verbose) std::cout << "collect rgb" << std::endl;
    rig.CollectRGB(gui, rgb) ;
    cuRgb.CopyFrom(rgb);
    if (gui.verbose) std::cout << "compute grey" << std::endl;
    tdp::Rgb2Grey(cuRgb,cuGreyFl,1./255.);

    cuPyrGreyFlPrev.CopyFrom(cuPyrGreyFlSmooth);
    cuGreyFlSmooth = cuPyrGreyFlSmooth.GetImage(0);
    if (smoothGrey==2) {
      tdp::Blur9(cuGreyFl,cuGreyFlSmooth, 1.);
    } else if (smoothGrey==1) {
      tdp::Blur5(cuGreyFl,cuGreyFlSmooth, 1.);
    } else {
      cuGreyFlSmooth.CopyFrom(cuGreyFl);
    }
    if (smoothGreyPyr==1) {
      tdp::CompletePyramidBlur9(cuPyrGreyFlSmooth, 1.);
    } else {
      tdp::CompletePyramidBlur(cuPyrGreyFlSmooth, 1.);
    }
    pyrGreyFlPrev.CopyFrom(pyrGreyFl);
    pyrGreyFl.CopyFrom(cuPyrGreyFlSmooth);
    greyFl = pyrGreyFl.GetImage(0);

    cuGradGrey = cuPyrGradGrey.GetImage(0);
    tdp::GradientShar(cuGreyFlSmooth, cuGradGrey);
    if (smoothGreyPyr==1) {
      tdp::CompletePyramidBlur9(cuPyrGradGrey, 1.);
    } else {
      tdp::CompletePyramidBlur(cuPyrGradGrey, 1.);
    }
    pyrGradGrey.CopyFrom(cuPyrGradGrey);
    gradGrey = pyrGradGrey.GetImage(0);

    tdp::Gradient2AngleNorm(cuGradGrey, cuGreyGradTheta, cuGreyGradNorm);
    greyGradNorm.CopyFrom(cuGreyGradNorm);

    pyrN.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    trackingGood = false;
    if (frame > 0 && runTracking && !gui.finished()) { // tracking
      if (doSO3prealign) {
        tdp::SO3f R_cp;
        if (gui.verbose) std::cout << "SO3 prealignment" << std::endl;
        if (useGpuPrealign) {
          TICK("icp RGB GPU");
          std::vector<size_t> maxIt(PYR, 0);
          for (int32_t pyr=SO3maxLvl; pyr>=SO3minLvl; --pyr) {
            maxIt[pyr] = SO3maxIt*(pyr-SO3minLvl)+1;
          }
          tdp::PhotometricSO3::ComputeProjective(cuPyrGreyFlPrev,
              cuPyrGreyFlSmooth, cuPyrGradGrey, cuPyrRay, cam, maxIt,
              gui.verbose, R_cp);
          TOCK("icp RGB GPU");
        } else {
          TICK("icp RGB");
          Eigen::Matrix<float,3,3> A;
          Eigen::Matrix<float,3,1> b;
          Eigen::Matrix<float,3,1> Ai;
          for (int32_t pyr=SO3maxLvl; pyr>=SO3minLvl; --pyr) {
            float scale = pow(0.5,pyr);
            CameraT camLvl = cam.Scale(scale);
            tdp::Image<float> greyFlLvl = pyrGreyFl.GetImage(pyr);
            tdp::Image<float> greyFlPrevLvl = pyrGreyFlPrev.GetImage(pyr);
            tdp::Image<tdp::Vector2fda> gradGreyLvl = pyrGradGrey.GetImage(pyr);
            tdp::Image<tdp::Vector3fda> rayLvl = pyrRay.GetImage(pyr);
            if (gui.verbose) std::cout << "pyramid lvl " << pyr << " scale " << scale << std::endl;
            for (size_t it = 0; it < SO3maxIt*(pyr-SO3minLvl)+1; ++it) {
              numInl = 0;
  //            for (auto& ass : assoc) mask[ass.second] = 0;
  //            assoc.clear();
              A = Eigen::Matrix<float,3,3>::Zero();
              b = Eigen::Matrix<float,3,1>::Zero();
              Ai = Eigen::Matrix<float,3,1>::Zero();
              float err = 0.;
  //            float H = 1e10;
              for (int32_t uP=0; uP<floor(w*scale); ++uP) {
                for (int32_t vP=0; vP<floor(h*scale); ++vP) {
  //                std::cout << uP << "," << vP << " " << floor(w*scale) << " " << floor(h*scale) << " "
  //                  << rayLvl.Description() << std::endl;
                  tdp::Vector2fda x = camLvl.Project(R_cp*rayLvl(uP,vP));
                  if (gradGreyLvl.Inside(x)) {
                    AccumulatePhotoSO3only(R_cp, camLvl, greyFlLvl.GetBilinear(x),
                        gradGreyLvl.GetBilinear(x),
                        rayLvl(uP,vP), greyFlPrevLvl(uP,vP), A, Ai, b, err);
                    numInl = numInl +1;
                  }
                }
              }
              // solve for x using ldlt
              Eigen::Matrix<float,3,1> x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
              R_cp = R_cp * tdp::SO3f::Exp_(scale*x);
  //            bool term = (x.norm()*180./M_PI < icpdRThr
  //                && tdp::CheckEntropyTermination(A, Hprev, SO3HThr, 0.f,
  //                  SO3negLogEvThr, H, gui.verbose));
              if (gui.verbose) {
                std::cout << "\tit " << it << ": err=" << err 
  //                << "\tH: " << H 
                  << "\t# inliers: " << numInl
                  << "\t|x|: " << x.norm()*180./M_PI << std::endl;
              }
  //            if (term) break;
            }
          }
          TOCK("icp RGB");
        }
        T_wc.rotation() = T_wc.rotation() * R_cp.Inverse();
      }
      pyrMask.Fill(0);
      if (runICP) {
        if (gui.verbose) std::cout << "SE3 ICP" << std::endl;
        TICK("icp");
        Eigen::Matrix<float,6,6> A;
        Eigen::Matrix<float,6,1> b;
        Eigen::Matrix<float,6,1> Ai;
        std::vector<uint32_t> maxItLvl = {maxIt0, maxIt1, maxIt2, maxIt3};
        for (int32_t pyr=frame==1? 0 : ICPmaxLvl; pyr>=0; --pyr) {
          std::vector<size_t> indK(invInd[pyr]->size(),0);
          float dotThr = cos(angleThr*M_PI/180.);
          float scale = pow(0.5,pyr);
          CameraT camLvl = cam.Scale(scale);
          tdp::Image<float> dLvl = pyrD.GetImage(pyr);
          tdp::Image<float> greyFlLvl = pyrGreyFl.GetImage(pyr);
          tdp::Image<float> greyFlPrevLvl = pyrGreyFlPrev.GetImage(pyr);
          tdp::Image<tdp::Vector2fda> gradGreyLvl = pyrGradGrey.GetImage(pyr);
          tdp::Image<tdp::Vector3fda> pcLvl = pyrPc.GetImage(pyr);
          tdp::Image<tdp::Vector3fda> nLvl = pyrN.GetImage(pyr);
          if (gui.verbose) std::cout << "pyramid lvl " << pyr << " scale " << scale << std::endl;
          for (size_t it = 0; it < maxItLvl[pyr]; ++it) {
            for (auto& ass : assoc) mask[ass.second] = 0;
            assoc.clear();
            indK = std::vector<size_t>(invInd[pyr]->size(),0);
            numProjected = 0;
            A = Eigen::Matrix<float,6,6>::Zero();
            b = Eigen::Matrix<float,6,1>::Zero();
            Ai = Eigen::Matrix<float,6,1>::Zero();
            float err = 0.;
            float H = 1e10;
            float Hprev = 1e10;
            tdp::SE3f T_cw = T_wc.Inverse();
            // associate new data until enough
            bool exploredAll = false;
            uint32_t k = 0;
            while (assoc.size() < 3000 && !exploredAll) {
              k = (k+1) % invInd[pyr]->size();
              while (indK[k] < invInd[pyr]->at(k).size()) {
                size_t i = invInd[pyr]->at(k)[indK[k]++];
                tdp::Plane& pl = pl_w.GetCircular(i);
                if (!pl.valid_)
                  continue;
                tdp::Vector3fda pc_w_in_c = T_cw*pl.p_;
                Eigen::Vector2f x = camLvl.Project(pc_w_in_c);
//                std::cout << x.transpose() << std::endl;
                if (!dLvl.Inside(x)) 
                  continue;
                float d_c = dLvl.GetBilinear(x(0),x(1));
//                std::cout << d_c << std::endl;
                if (d_c != d_c || fabs(d_c-pc_w_in_c(2)) > occlusionDepthThr) 
                  continue;
                numProjected = numProjected + 1;
                int32_t u = floor(x(0)+0.5f);
                int32_t v = floor(x(1)+0.5f);
                if (!EnsureNormal(pcLvl, dpc, W, nLvl, curv, u, v, normalViaRMLS))
                  //              if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, camLvl, pc,
                  //                    W, dpc, n, curv, u,v ))
                  continue;
                if (useTexture) {
                  if (!AccumulateP2PlIntensity(pl, T_wc, T_cw, camLvl, pcLvl(u,v),
                        nLvl(u,v), greyFlLvl(u,v), gradGreyLvl(u,v), p2plThr, dotThr,
                        lambdaTex, A, Ai, b, err))
                    continue;
                } else if (use3dGrads) {
                  if (!AccumulateP2Pl3DGrad(pl, T_wc, T_cw, camLvl, pcLvl(u,v),
                        nLvl(u,v), greyFlLvl(u,v), gradGreyLvl(u,v),
                        x(0),x(1), p2plThr, dotThr, lambdaTex, A, Ai, b,
                        err))
                    continue;
                } else if (useNormals) {
                  tdp::Vector3fda n_wi = pl.n_;
                  if (usevMFmeans) {
                    std::lock_guard<std::mutex> lock(vmfsLock);
                    n_wi = vmfs[pl.z_].mu_;
                  }
                  if (!AccumulateP2PlNormal(pl, n_wi, T_wc, T_cw, camLvl, pcLvl(u,v),
                        nLvl(u,v), p2plThr, dotThr, lambdaNs, A, Ai, b, err)) {
                    continue;
                  }
                } else if (useNormalsAndTexture) {
                  tdp::Vector3fda n_wi = pl.n_;
                  if (usevMFmeans) {
                    std::lock_guard<std::mutex> lock(vmfsLock);
                    n_wi = vmfs[pl.z_].mu_;
                  }
                  if (!AccumulateP2PlIntensityNormals(pl, n_wi, T_wc, T_cw, camLvl, pcLvl(u,v),
                        nLvl(u,v), greyFlLvl(u,v),gradGreyLvl(u,v), p2plThr, dotThr,
                        lambdaNs, lambdaTex, A, Ai, b, err)) {
                    continue;
                  }
                } else {
                  if (!AccumulateP2Pl(pl, T_wc, T_cw, pcLvl(u,v), nLvl(u,v),
                        p2plThr, dotThr, A, Ai, b, err))
                    continue;
                }
                mask(u,v) = 255;
                assoc.emplace_back(i,u+v*pc.w_);
                break;
              }
              if (k == 0) {
                if (tdp::CheckEntropyTermination(A, Hprev, HThr+pyr*dPyrHThr, condEntropyThr, 
                      negLogEvThr+pyr*dPyrNewLogEvHThr, H, gui.verbose))
                  break;
                Hprev = H;
              }
              exploredAll = true;
              for (size_t k=0; k<indK.size(); ++k) exploredAll &= indK[k] >= invInd[pyr]->at(k).size();
            }
            numInl = assoc.size();
            Eigen::Matrix<float,6,1> x = Eigen::Matrix<float,6,1>::Zero();
            if (assoc.size() > 6) { // solve for x using ldlt
              //            std::cout << "A: " << std::endl << A << std::endl << "b: " << b.transpose() << std::endl;
              x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
              T_wc = T_wc * tdp::SE3f::Exp_(x*pow(dPyrdAlpha,pyr));
            }
            if (gui.verbose) {
              std::cout << "\tit " << it << ": err=" << err 
                << "\t# inliers: " << numInl
                << "\t|x|: " << x.topRows(3).norm()*180./M_PI 
                << " " <<  x.bottomRows(3).norm() << std::endl;
            }
            if (x.topRows<3>().norm()*180./M_PI < icpdRThr
                && x.bottomRows<3>().norm() < icpdtThr
                && tdp::CheckEntropyTermination(A, Hprev, HThr+pyr*dPyrHThr, 0.f,
                  negLogEvThr+pyr*dPyrNewLogEvHThr, H, gui.verbose)) {
              break;
            }
          }
//          if (gui.verbose) {
//            for (size_t k=0; k<invInd[pyr]->size(); ++k) {
//              if (invInd[pyr]->at(k).size() > 0 )
//                std::cout << "used different directions " << k << "/" 
//                  << invInd[pyr]->size() << ": " << indK[k] 
//                  << " of " << invInd[pyr]->at(k).size() << std::endl;
//            }
//          }
        }
        logObs.Log(log(assoc.size())/log(10.), 
            log(numProjected)/log(10.), log(pl_w.SizeToRead())/log(10));
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
        Eigen::Matrix<float,6,1> ev = eig.eigenvalues().real();
        Eigen::Matrix<float,6,6> Q = eig.eigenvectors();
        float H = -ev.array().log().sum();
        if (gui.verbose) {
          std::cout << " H " << H << " neg log evs " << 
            -ev.array().log().matrix().transpose() << std::endl;
        }

        //for (size_t k=0; k<K; ++k) {
        //  Eigen::Matrix<float,6,1> Ai;
        //  Ai << Eigen::Vector3f::Zero(), dpvmf.GetCenter(k);
        //  std::cout << "k " << k << std::endl;
        //  std::cout << (Q.transpose()*Ai*Ai.transpose()*Q).diagonal().transpose() << std::endl;
        //}

        logEntropy.Log(H);
        logEig.Log(-ev.array().log().matrix());
        Eigen::Matrix<float,6,1> q0 = Q.col(0);
        uint32_t maxId = 0;
        q0.array().abs().maxCoeff(&maxId);
        q0 *= (q0(maxId) > 0? 1.: -1.);
        logEv.Log(q0);
        T_wcs.push_back(T_wc);
        trackingGood = (H <= HThr && assoc.size() > 10) || assoc.size() > 0.5*idsCur[0]->size();
        TOCK("icp");
        if (trackingGood) if (gui.verbose) std::cout << "tracking good" << std::endl;
      }

      if (trackingGood) {
        std::lock_guard<std::mutex> mapGuard(mapLock);
        TICK("update planes");
//        size_t numNN = 0;
//        tdp::SE3f T_cw = T_wc.Inverse();
        for (const auto& ass : assoc) {
          size_t i = ass.first;
          tdp::Plane& pl = pl_w[i];
          if (!pl.valid_)
            continue;

          int32_t u = ass.second%pc.w_;
          int32_t v = ass.second/pc.w_;

          tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);
          tdp::Vector3fda n_c_in_w = T_wc.rotation()*n(u,v);

            tdp::Matrix3fda SigmaO;
            NoiseModelNguyen(n(u,v), pc(u,v), cam, SigmaO);
            float sigmaSqO = 100*n(u,v).dot(SigmaO*n(u,v));

          ts[i] = frame;
          pl.lastFrame_ = frame;
          pl.numObs_ ++;

          float w = numSum_w[i];
          // filtering grad grey
          pl.grad_ = (pl.grad_*w + pl.Compute3DGradient(T_wc, cam, u, v, gradGrey(u,v)))/(w+1);
          pl.grey_ = (pl.grey_*w + greyFl(u,v)) / (w+1);
          pl.gradNorm_ = (pl.gradNorm_*w + gradGrey(u,v).norm()) / (w+1);
          pl.rgb_ = ((pl.rgb_.cast<float>()*w + rgb(u,v).cast<float>()) / (w+1)).cast<uint8_t>();
          outerSum_w[i] = (outerSum_w[i]*w + pc_c_in_w*pc_c_in_w.transpose())/(w+1.);

          tdp::Matrix3fda infoObs = (T_wc.rotation().matrix().transpose()*SigmaO*T_wc.rotation().matrix().transpose()).inverse();
          tdp::Vector3fda xiObs = infoObs*pc_c_in_w;
          pcObsInfo_w[i] = (pcObsInfo_w[i]*w + infoObs)/(w+1.);
          pcObsXi_w[i] = (pcObsXi_w[i]*w + xiObs)/(w+1.);

//          pcSum_w[i] = (pcSum_w[i]*w + pc_c_in_w/(sigmaObsP*sigmaObsP))/(w+1.);
//          infoObsSum[i] = (infoObsSum[i]*w + 1./(sigmaObsP*sigmaObsP))/(w+1.);
            pcSum_w[i] = (pcSum_w[i]*w + pc_c_in_w/sigmaSqO)/(w+1.);
            infoObsSum[i] = (infoObsSum[i]*w + 1./sigmaSqO)/(w+1.);

          nSum_w[i] = (nSum_w[i]*w + n_c_in_w)/(w+1.);
          normSum_w[i] = nSum_w[i].norm();
          nSum_w[i] /= nSum_w[i].norm();
          numSum_w[i] = std::min(50.f, w+1.f);

          pcEst_w[i] = pcSum_w[i]/infoObsSum[i];
          grad_w[i] = pl.grad_;
          gradDir_w[i] = grad_w[i].normalized();

//          if (updateMap) {
//            for (size_t i=0; i<kNN; ++ i) {
//              for (const auto& assB : assoc) {
//                if (assB.first == nn[ass.first](i)){
//                  int32_t uB = assB.second%pc.w_;
//                  int32_t vB = assB.second/pc.w_;
//                  //mapObsP2Pl[ass.first](i) += n(u,v).dot(pc(uB,vB)-pc(u,v));
//                  //mapObsDot[ass.first](i) += n(u,v).dot(n(uB,vB));
//                  //mapObsNum[ass.first](i) ++;
//                  float w = mapObsNum[ass.first](i);
//                  mapObsP2Pl[ass.first](i) = (w*mapObsP2Pl[ass.first](i) + n(u,v).dot(pc(uB,vB)-pc(u,v)))/(w+1.);
//                  mapObsDot[ass.first](i) = (mapObsDot[ass.first](i)*w + n(u,v).dot(n(uB,vB)))/(w+1.);
//                  mapObsNum[ass.first](i) = std::min(100.f,w+1.f) ;
//                  //std::cout << "found NN " << i << " of " << ass.first 
//                  //  << " " << mapObsP2Pl[ass.first](i)/mapObsNum[ass.first](i) 
//                  //  << " " << mapObsDot[ass.first](i)/mapObsNum[ass.first](i)
//                  //  << " " << mapObsNum[ass.first](i) << std::endl;
//                  numNN++;
//                  break;
//                }
//              }
//            }
          if (!updateMap) {
            pl.AddObs(pc_c_in_w, n_c_in_w);
            n_w[i] =  pl.n_;
            pc_w[i] = pl.p_;
          }
        }
        uint32_t j = 0;
        for (auto& i : *idsCur[0]) {
          if (pl_w[i].lastFrame_ < frame && j++ < numAdditionalObs) {
            tdp::Plane& pl = pl_w[i];
            if (!pl.valid_)
              continue;
            tdp::Vector3fda pc_w_in_c = T_wc.Inverse()*pl.p_;
            Eigen::Vector2f x = cam.Project(pc_w_in_c);
            if (!d.Inside(x)) 
              continue;
            int32_t u = floor(x(0)+0.5f);
            int32_t v = floor(x(1)+0.5f);
            float d_c = d(u,v);
            if (d_c != d_c || fabs(d_c-pc_w_in_c(2)) > occlusionDepthThr) 
              continue;
            if (!EnsureNormal(pc, dpc, W, n, curv, u, v, normalViaRMLS))
              continue;
//            std::cout << " adding " << j << " of " << numAdditionalObs 
//              << " id " << i << " uv " << u << "," << v << std::endl;
            numProjected = numProjected + 1;
            tdp::Vector3fda n_c_in_w = T_wc.rotation()*n(u,v);
            tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);

            tdp::Matrix3fda SigmaO;
            NoiseModelNguyen(n(u,v), pc(u,v), cam, SigmaO);
            float sigmaSqO =100* n(u,v).dot(SigmaO*n(u,v));

//            std::cout << sigmaSqO << std::endl;

            // TODO: copied from above
            ts[i] = frame;
            pl.lastFrame_ = frame;
            pl.numObs_ ++;
            float w = numSum_w[i];
            // filtering grad grey
            pl.grad_ = (pl.grad_*w + pl.Compute3DGradient(T_wc, cam, u, v, gradGrey(u,v)))/(w+1);
            pl.grey_ = (pl.grey_*w + greyFl(u,v)) / (w+1);
            pl.gradNorm_ = (pl.gradNorm_*w + gradGrey(u,v).norm()) / (w+1);
            pl.rgb_ = ((pl.rgb_.cast<float>()*w + rgb(u,v).cast<float>()) / (w+1)).cast<uint8_t>();
            outerSum_w[i] = (outerSum_w[i]*w + pc_c_in_w*pc_c_in_w.transpose())/(w+1.);

            tdp::Matrix3fda infoObs = (T_wc.rotation().matrix().transpose()*SigmaO*T_wc.rotation().matrix().transpose()).inverse();
            tdp::Vector3fda xiObs = infoObs*pc_c_in_w;
            pcObsInfo_w[i] = (pcObsInfo_w[i]*w + infoObs)/(w+1.);
            pcObsXi_w[i] = (pcObsXi_w[i]*w + xiObs)/(w+1.);

            pcSum_w[i] = (pcSum_w[i]*w + pc_c_in_w/sigmaSqO)/(w+1.);
            infoObsSum[i] = (infoObsSum[i]*w + 1./sigmaSqO)/(w+1.);
//            pcSum_w[i] = (pcSum_w[i]*w + pc_c_in_w/(sigmaObsP*sigmaObsP))/(w+1.);
//            infoObsSum[i] = (infoObsSum[i]*w + 1./(sigmaObsP*sigmaObsP))/(w+1.);

            nSum_w[i] = (nSum_w[i]*w + n_c_in_w)/(w+1.);
            normSum_w[i] = nSum_w[i].norm();
            nSum_w[i] /= nSum_w[i].norm();
            numSum_w[i] = std::min(50.f, w+1.f);

            pcEst_w[i] = pcSum_w[i]/infoObsSum[i];
            grad_w[i] = pl.grad_;
            gradDir_w[i] = grad_w[i].normalized();

//            if (j++ >= numAdditionalObs) 
//              break;
          }
        }
        
//        if (gui.verbose) std::cout << "num NN measured " << numNN << std::endl;
        TOCK("update planes");
      }
    }

//    if (runLoopClosureGeom && K>2) {
//      tdp::ManagedDPvMFmeansSimple3fda dpvmfCur(35.*M_PI/180.);
//      for (const auto& ass : assoc) {
//        dpvmfCur.addObservation(n(ass.second%pc.w_,ass.second/pc.w_));
//      }
//      dpvmfCur.iterateToConvergence(100, 1e-6);
//      if (dpvmfCur.GetK() > 2) {
//        std::vector<size_t> idsW(K);
//        std::vector<size_t> idsC(dpvmfCur.GetK());
//        std::iota(idsW.begin(), idsW.end(), 0);
//        std::iota(idsC.begin(), idsC.end(), 0);
//        Eigen::Matrix3f N;
//        float maxAlign = 0;
//        for (size_t it =0; it < 1000; ++it) {
//          std::random_shuffle(idsW.begin(), idsW.end());
//          std::random_shuffle(idsC.begin(), idsC.end());
//          N = Eigen::Matrix3f::Zero();
//          for (size_t i=0; i<3; ++i) {
//            N += dpvmf.GetCenter(idsW[i]) * dpvmfCur.GetCenter(idsC[i]).transpose();
//          }
//          // TODO check order
//          Eigen::Matrix3f R_wc = tdp::ProjectOntoSO3<float>(N);
//          float align = (R_wc*N).trace();
//          if (align > maxAlign) {
//            T_wcRansac.rotation() = tdp::SO3f(R_wc);
//          }
//        }
//      }
//    }

    frame ++;

    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    TICK("Draw 3D");

    if (showPcCurrent) {
      TICK("Draw 3D vbo cbo upload");
      if (showPcLvl == 0) {
        vbo.Reinitialise(pangolin::GlArrayBuffer,pc.Area(),GL_FLOAT,3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
        cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
      } else {
        vbo.Reinitialise(pangolin::GlArrayBuffer,pyrPc.GetImage(showPcLvl).Area(),GL_FLOAT,3, GL_DYNAMIC_DRAW);
        vbo.Upload(pyrPc.GetImage(showPcLvl).ptr_, pyrPc.GetImage(showPcLvl).SizeBytes(), 0);
      }
      TOCK("Draw 3D vbo cbo upload");
    }

    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);
      glClearColor(bgGrey, bgGrey, bgGrey, 1.0f);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

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

      TICK("Draw 3D nbo upload");
      if (showSamples) {
        nbo_w.Upload(nS.ptr_, n_w.SizeToReadBytes(), 0);
      } else {
        nbo_w.Upload(n_w.ptr_, n_w.SizeToReadBytes(), 0);
      }
      TOCK("Draw 3D nbo upload");

      if (showFullPc) {
        TICK("Draw 3D render PC");
        // TODO I should not need to upload all of pc_w everytime;
        // might break things though
        vbo_w.Upload(pc_w.ptr_, pc_w.SizeToReadBytes(), 0);
        cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeToReadBytes(), 0);
//        cbo_w.Upload(&rgb_w.ptr_[iReadCurW], 
//            rgb_w.SizeToRead(iReadCurW)*sizeof(tdp::Vector3fda), 
//            iReadCurW*sizeof(tdp::Vector3fda));
        pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
        pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
        if (showAge || showObs || showCurv || showGrey || showNumSum || showZCounts
            || showHn || showHp || showInfoObs) {
          if (showAge) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = ts.GetCircular(i);
          } else if (showObs) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).numObs_;
          } else if (showGrey) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).grey_;
          } else if (showNumSum) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = numSum_w[i];
          } else if (showZCounts) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = zCountS[i];
          } else if (showHp) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w[i].Hp_;
          } else if (showHn) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w[i].Hn_;
          } else if (showInfoObs) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = infoObsSum[i];
          } else {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).curvature_;
          }
          valuebo.Upload(age.ptr_, pl_w.SizeToRead()*sizeof(float), 0);
          std::pair<float,float> minMaxAge = age.GetRoi(0,0,
              pl_w.SizeToRead(),1).MinMax();
          std::cout << "drawn values are min " << minMaxAge.first << " max " << minMaxAge.second << std::endl;
          tdp::RenderVboValuebo(vbo_w, valuebo, minMaxAge.first,
              minMaxAge.second, P, MV);
        } else if (showLabels && frame > 1) {
          lbo.Upload(zS.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
          tdp::RenderLabeledVbo(vbo_w, lbo, s_cam);
        } else if (showSurfels) {
          if (gui.verbose) std::cout << "render surfels" << std::endl;
          tdp::RenderSurfels(vbo_w, nbo_w, cbo_w, rbo, dMax, P, MV);
        } else {
          pangolin::RenderVboCbo(vbo_w, cbo_w, true);
        }
        if (showNN) {
          TICK("Draw 3D render NN");
//          std::cout << pl_w.SizeToRead() << " vs " << mapNN.size() << " -> "
//             << mapNN.size()/kNN << std::endl;
//          for (auto& ass : mapNN) {
          for (size_t i=0; i<mapNN.size(); ++i) {
            auto& ass = mapNN[i];
            if (ass.second >= 0) {
              if (numSamplesZ[ass.first](i%kNN) > 0) {
                tdp::glColorHot(sumSameZ[ass.first](i%5)/numSamplesZ[ass.first](i%5));
              } else {
                glColor4f(0.3,0.3,0.3,0.3);
              }
              tdp::glDrawLine(pl_w[ass.first].p_, pl_w[ass.second].p_);
            }
          }
          TOCK("Draw 3D render NN");
        }
        TOCK("Draw 3D render PC");
      }

      if (showNormals) {
        TICK("Draw 3D render normals");
        tdp::ShowCurrentNormals(pc, n, assoc, T_wc, scale);
        glColor4f(0,1,0,0.5);
        for (size_t i=0; i<n_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w.GetCircular(i), 
              pc_w.GetCircular(i) + scale*n_w.GetCircular(i));
        }
        TOCK("Draw 3D render normals");
      } else if (showGrads) {
        TICK("Draw 3D render grads");
        glColor4f(0,1,0,0.5);
        for (size_t i=0; i<grad_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w.GetCircular(i), 
              pc_w.GetCircular(i) + 10.*scale*grad_w.GetCircular(i));
        }
        TOCK("Draw 3D render grads");
      }

      // render current camera second in the propper frame of
      // reference
      if (showPcCurrent) {
        pangolin::glSetFrameOfReference(T_wc.matrix());
        pangolin::RenderVboCbo(vbo, cbo, true);
        pangolin::glUnsetFrameOfReference();
      }
      if (showPcEst) {
        vboEst_w.Upload(pcEst_w.ptr_, pl_w.SizeToRead()*sizeof(tdp::Vector3fda), 0);
        glColor3f(0.,1.,1.);
        pangolin::RenderVbo(vboEst_w);
        glColor4f(1,0,1,0.5);
        for (size_t i=0; i<pl_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w[i], pcEst_w[i]);
        }
      }
    }

    if (viewAssoc.IsShown()) {
      viewAssoc.Activate(s_cam);
      pangolin::glSetFrameOfReference(T_wc.matrix());
      pangolin::glDrawAxis(0.1f);
      if (showPcCurrent) {
        if (showPcLvl == 0) {
          pangolin::RenderVboCbo(vbo, cbo, true);
        } else {
          glColor3f(0,1,1);
          pangolin::RenderVbo(vbo);
        }
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
      if (frame > 1) {
        lbo.Upload(zS.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
        tdp::RenderLabeledVbo(nbo_w, lbo, normalsCam);
      }  
      glColor4f(0,1,0,1.);
      {
        std::lock_guard<std::mutex> lock(vmfsLock);
        for (size_t k=0; k<vmfs.size(); ++k) {
          if (vmfSS[k](3) > 0) {
            tdp::glDrawLine(tdp::Vector3fda::Zero(), vmfs[k].mu_);
            std::cout << "vmf " << k << " tau: " << vmfs[k].tau_ << std::endl;
          }
        }
      }
    }
    if (viewGrads.IsShown()) {
      Eigen::Matrix4f Tview = s_cam.GetModelViewMatrix();
      Tview(0,3) = 0.; Tview(1,3) = 0.; Tview(2,3) = -2.2;
      normalsCam.GetModelViewMatrix() = Tview;
      viewGrads.Activate(normalsCam);
      glColor4f(0,0,1,0.5);
      if (showGradDir) {
        gradbo_w.Upload(gradDir_w.ptr_, pl_w.SizeToRead()*sizeof(tdp::Vector3fda), 0);
      } else {
        gradbo_w.Upload(grad_w.ptr_, pl_w.SizeToRead()*sizeof(tdp::Vector3fda), 0);
      }
      pangolin::RenderVbo(gradbo_w);
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
      for (auto& ass : assoc) {
        pangolin::glDrawCircle(ass.second%wc,ass.second/wc,1);
      }
      glColor3f(0,1,0);
      for (auto& id : idNew) {
        pangolin::glDrawCircle(id%wc,id/wc,1);
      }
    }

    if (containerTracking.IsShown()) {
      if (viewGrey.IsShown()) {
        tdp::PyramidToImage(pyrGreyFl, pyrGreyFlImg);
        viewGrey.SetImage(pyrGreyFlImg);
      }
      if (viewGradGrey.IsShown()) {
        tdp::PyramidToImage(cuPyrGradGrey, cuGrad2D);
        tdp::Grad2Image(cuGrad2D, cuGrad2DImg);
        grad2DImg.CopyFrom(cuGrad2DImg);
        viewGradGrey.SetImage(grad2DImg);
      }
      if (viewD.IsShown()) {
        tdp::PyramidToImage(pyrD, pyrDImg);
        viewD.SetImage(pyrDImg);
      }
      if (viewMask.IsShown()) {
        tdp::PyramidToImage(pyrMaskDisp, pyrMaskImg);
        viewMask.SetImage(pyrMaskImg);
      }
    }
    if (!gui.finished() && plotters.IsShown()) {
      plotdH.ScrollView(1,0);
      plotH.ScrollView(1,0);
      plotObs.ScrollView(1,0);
      plotEig.ScrollView(1,0);
      plotEv.ScrollView(1,0);
    }

    TOCK("Draw 2D");
    if (record) {
      std::string name = tdp::MakeUniqueFilename("sparseFusion.png");
      name = std::string(name.begin(), name.end()-4);
      gui.container().SaveOnRender(name);
    }

    if (gui.verbose) std::cout << "finished one iteration" << std::endl;
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();

    if (!gui.finished()) {
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
  }
  out.close();

  for (size_t lvl=0; lvl<PYR; ++lvl) {
    delete idsCur[lvl];
    delete invInd[lvl];
  }

//  imuInterp.Stop();
//  if (imu) imu->Stop();
//  delete imu;
//  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

