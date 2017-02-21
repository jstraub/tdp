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
    ManagedHostCircularBuffer<Vector3fda>& grad_w,
    ManagedHostCircularBuffer<float>& rs,
    ManagedHostCircularBuffer<uint16_t>& ts
    ) {
  Plane pl;
  tdp::Brief feat;
  Vector3fda n, p;
  float curv;
  for (size_t i=0; i<mask.Area(); ++i) {
    if (mask[i] 
        && tdp::IsValidData(pc[i]) ) {
//      uint32_t Wscaled = floor(W*pc[i](2));
      uint32_t Wscaled = W;
      const uint32_t u = i%mask.w_;
      const uint32_t v = i/mask.w_;
  
//      if (tdp::NormalViaScatter(pc, i%mask.w_, i/mask.w_, Wscaled, n)) {
      if (tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, 
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

        pl.grad_ = pl.Compute3DGradient(T_wc, cam, u, v, gradGrey[i]);
//        float uGrad = u + 10.*pl.gradGrey_(0);
//        float vGrad = v + 10.*pl.gradGrey_(1);
//        tdp::Rayfda ray(tdp::Vector3fda::Zero(),
//            cam.Unproject(uGrad,vGrad,1.));
//        ray.Transform(T_wc);
//        pl.grad_ = pl.gradGrey_.norm()*(ray.IntersectPlane(pl.p_,pl.n_) - pl.p_).normalized();
        // could project onto plane spanned by normal?

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
bool AccumulateP2PlIntensity(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
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
  tdp::Vector3fda pc_w_in_c = T_cw*pc_w;
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
//        Ai.bottomRows<3>() = n_w; 
        bi = p2pl;
        A += Ai * Ai.transpose();
        b += Ai * bi;
        err += bi;
//        std::cout << " p2pl " << bi << " " << Ai.transpose() << std::endl;
        // texture old
//        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_c_in_w);
//        Eigen::Matrix<float,3,6> Jse3;
//        Jse3 << -(T_wc.rotation().matrix()*SO3mat<float>::invVee(pc_ci)), 
//             Eigen::Matrix3f::Identity();
//        Ai = Jse3.transpose() * Jpi.transpose() * pl.gradGrey_;
        // texture 3D grads
//        if (tdp::IsValidData(pl.grad_)) {
//          Eigen::Matrix<float,3,6> Jse3;
//          Jse3.leftCols<3>() = -(T_wc.rotation().matrix()
//              *SO3mat<float>::invVee(pc_ci));
//          Jse3.rightCols<3>() = Eigen::Matrix3f::Identity();
//          Ai = Jse3.transpose() * pl.grad_;
//          bi = grey_ci - pl.grey_;
//          A += lambda*(Ai * Ai.transpose());
//          b += lambda*(Ai * bi);
//          err += lambda*bi;
//        } else {
//          std::cout << " grad 3D is nan!" << std::endl; 
//        }
        // texture inverse transform verified Jse3 
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_w_in_c);
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
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float distThr, 
    float p2plThr, 
    float dotThr,
    float gamma,
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

/// uses gradient and normal as well
bool AccumulateP2PlIntensityNormals(const Plane& pl, 
    tdp::SE3f& T_wc, 
    tdp::SE3f& T_cw, 
    CameraT& cam,
    const Vector3fda& pc_ci,
    const Vector3fda& n_ci,
    float grey_ci,
    const Vector2fda& gradGrey_ci,
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
  tdp::Vector3fda pc_w_in_c = T_cw*pc_w;
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
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(pc_w_in_c);
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
  tdp::ManagedHostPyramid<float,3> pyrGreyFl(wc,hc);
  tdp::Image<float> greyFl = pyrGreyFl.GetImage(0);
  tdp::ManagedDeviceImage<uint8_t> cuGrey(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreyFl(wc,hc);
  tdp::ManagedHostImage<float> pyrGreyFlImg(3*wc/2, hc); 
  tdp::ManagedDevicePyramid<float,3> cuPyrGreyFlSmooth(wc,hc);
  tdp::Image<float> cuGreyFlSmooth = cuPyrGreyFlSmooth.GetImage(0);
  tdp::ManagedDeviceImage<float> cuGreyDu(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDv(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyGradNorm(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyGradTheta(wc,hc);
  tdp::ManagedHostImage<float> greyGradNorm(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector2fda,3> cuPyrGradGrey(wc,hc);
  tdp::ManagedHostPyramid<tdp::Vector2fda,3> pyrGradGrey(wc,hc);
  tdp::Image<tdp::Vector2fda> cuGradGrey = cuPyrGradGrey.GetImage(0);
  tdp::Image<tdp::Vector2fda> gradGrey = pyrGradGrey.GetImage(0);

  tdp::ManagedDeviceImage<tdp::Vector2fda> cuGrad2D(3*wc/2, hc); 
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuGrad2DImg(3*wc/2, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> grad2DImg(3*wc/2, hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedDeviceImage<uint8_t> cuMask(wc, hc);
  tdp::ManagedHostImage<uint8_t> mask(wc, hc);
  tdp::ManagedHostImage<uint32_t> z(w, h);

  tdp::ManagedHostImage<float> age(MAP_SIZE);


  // ICP stuff
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

//  tdp::ManagedHostImage<tdp::Vector3fda> pc_c;
//  tdp::ManagedHostImage<tdp::Vector3bda> rgb_c;
//  tdp::ManagedHostImage<tdp::Vector3fda> n_c;

  pangolin::Var<bool> record("ui.record",false,true);
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",6.,0.1,10.);

  pangolin::Var<float> subsample("ui.subsample %",1.,0.1,3.);
  pangolin::Var<float> pUniform("ui.p uniform ",0.1,0.1,1.);
  pangolin::Var<float> scale("ui.scale",0.05,0.1,1);
  pangolin::Var<float> bgGrey("ui.bg Grey",0.02,0.0,1);

  pangolin::Var<int> numMapPoints("ui.num Map",0,0,0);
  pangolin::Var<int> numProjected("ui.num Proj",0,0,0);
  pangolin::Var<int> numInl("ui.num Inl",0,0,0);
  pangolin::Var<int> idMapUpdate("ui.id Map",0,0,0);
  pangolin::Var<int> idNNUpdate("ui.id NN",0,0,0);

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> runLoopClosure("ui.run loop closure",false,true);
  pangolin::Var<bool> runLoopClosureGeom("ui.run loop closure geom",false,true);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updatePlanes("ui.update planes",true,true);
  pangolin::Var<bool> updateMap("ui.update map",true,true);
  pangolin::Var<float> occlusionDepthThr("ui.occlusion D Thr",0.3,0.01,0.3);

  pangolin::Var<int> smoothGrey("ui.smooth grey",1,0,2);
  pangolin::Var<int> smoothGreyPyr("ui.smooth grey pyr",1,0,1);
  pangolin::Var<bool> showGradDir("ui.showGradDir",true,true);

  pangolin::Var<bool> doRegvMF("ui.reg vMF",false,true);
  pangolin::Var<bool> doRegPc0("ui.reg pc0",false,true);
  pangolin::Var<bool> doRegAbsPc("ui.reg abs pc",true,true);
  pangolin::Var<bool> doRegAbsN("ui.reg abs n",true,true);
  pangolin::Var<bool> doRegRelPlZ("ui.reg rel Pl",true,true);
  pangolin::Var<bool> doRegRelNZ("ui.reg rel N",true,true);
  pangolin::Var<bool> doRegRelPlObs("ui.reg rel PlObs",false,true);
  pangolin::Var<bool> doRegRelNObs("ui.reg rel NObs",false,true);
  pangolin::Var<bool> doVariationalUpdate("ui.variational",false,true);
  pangolin::Var<float> lambdaRegDir("ui.lamb Reg Dir",0.01,0.01,1.);
  pangolin::Var<float> lambdaRegPl("ui.lamb Reg Pl",1.0,0.01,10.);
  pangolin::Var<float> lambdaRegPc0("ui.lamb Reg Pc0",0.01,0.01,1.);
  pangolin::Var<float> lambdaMRF("ui.lamb z MRF",.1,0.01,10.);
  pangolin::Var<float> alphaGrad("ui.alpha Grad",.0005,0.0,1.);

  pangolin::Var<bool> pruneAssocByRender("ui.prune assoc by render",true,true);
  pangolin::Var<int> dtAssoc("ui.dtAssoc",5000,1,1000);
  pangolin::Var<float> lambdaNs("ui.lamb Ns",0.01,0.001,1.);
  pangolin::Var<float> lambdaNsOld("ui.lamb Ns old",0.1,0.01,1.);
  pangolin::Var<float> lambdaTex("ui.lamb Tex",0.1,0.01,1.);
  pangolin::Var<bool> useTexture("ui.use Tex ICP",true,true);
  pangolin::Var<bool> useNormals("ui.use Ns ICP",false,true);
  pangolin::Var<bool> useNormalsAndTexture("ui.use Tex&Ns ICP",false,true);

  pangolin::Var<bool> runICP("ui.run ICP",true,true);
  pangolin::Var<bool> icpReset("ui.reset icp",true,false);
  pangolin::Var<float> angleUniformityThr("ui.angle unif thr",5, 0, 90);
  pangolin::Var<float> angleThr("ui.angle Thr",15, -1, 90);
//  pangolin::Var<float> angleThr("ui.angle Thr",-1, -1, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.03,0,0.3);
  pangolin::Var<float> distThr("ui.dist Thr",0.1,0,0.3);
  pangolin::Var<float> curvThr("ui.curv Thr",1.,0.01,1.0);
  pangolin::Var<float> assocDistThr("ui.assoc dist Thr",0.1,0,0.3);
  pangolin::Var<float> HThr("ui.H Thr",-32.,-40.,-12.);
  pangolin::Var<float> negLogEvThr("ui.neg log ev Thr",-4.,-12.,-1.);
  pangolin::Var<float> condEntropyThr("ui.rel log dH ", 1.e-3,1.e-3,1e-2);
  pangolin::Var<float> icpdRThr("ui.dR Thr",0.25,0.1,1.);
  pangolin::Var<float> icpdtThr("ui.dt Thr",0.01,0.01,0.001);
  pangolin::Var<int> numRotThr("ui.numRot Thr",200, 100, 350);
  pangolin::Var<int> maxIt("ui.max iter",15, 1, 20);

  pangolin::Var<bool> doSO3prealign("ui.SO3 prealign",true,true);
  pangolin::Var<float> SO3HThr("ui.SO3 H Thr",-24.,-40.,-20.);
  pangolin::Var<float> SO3negLogEvThr("ui.SO3 neg log ev Thr",-6.,-10.,0.);
  pangolin::Var<float> SO3condEntropyThr("ui.SO3 rel log dH ", 1.e-3,1.e-6,1e-2);
  pangolin::Var<int> SO3maxIt("ui.SO3 max iter",3, 1, 20);
  pangolin::Var<int> SO3maxLvl("ui.SO3 max Lvl",2,0,2);

  pangolin::Var<int>   W("ui.W ",9,1,15);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool> showPlanes("ui.show planes",false,true);
  pangolin::Var<bool> showPcModel("ui.show model",false,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",false,true);
  pangolin::Var<bool> showFullPc("ui.show full",true,true);
  pangolin::Var<bool> showNormals("ui.show ns",true,true);
  pangolin::Var<bool> showGrads("ui.show grads",true,true);
  pangolin::Var<bool> showAge("ui.show age",false,true);
  pangolin::Var<bool> showObs("ui.show # obs",false,true);
  pangolin::Var<bool> showCurv("ui.show curvature",false,true);
  pangolin::Var<bool> showGrey("ui.show grey",false,true);
  pangolin::Var<bool> showDPvMFlabels("ui.show DPvMF labels",true,true);
  pangolin::Var<bool> showLabels("ui.show labels",true,true);
  pangolin::Var<bool> showSamples("ui.show Samples",false,true);
  pangolin::Var<bool> showSurfels("ui.show surfels",true,true);
  pangolin::Var<bool> showNN("ui.show NN",false,true);
  pangolin::Var<bool> showLoopClose("ui.show loopClose",false,true);
  pangolin::Var<int> step("ui.step",10,1,100);

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

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> rs(MAP_SIZE); // radius of surfels
  tdp::ManagedHostCircularBuffer<uint16_t> ts(MAP_SIZE); // radius of surfels
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> grad_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> gradDir_w(MAP_SIZE);

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
  std::mutex dpvmfLock;
  std::thread topology([&]() {
    int32_t iReadNext = 0;
    int32_t sizeToRead = 0;
    tdp::VectorkNNfda values;
    while(runTopologyThread.Get()) {
      {
        std::lock_guard<std::mutex> lock(pl_wLock); 
        sizeToRead = pl_w.SizeToRead();
      }
      if (sizeToRead > 0) {
        values.fill(std::numeric_limits<float>::max());
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
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
        for (int32_t i=0; i<kNN; ++i) {
//            mapObsDot[iReadNext][i] = pl.n_.dot(pl_w[ids[i]].n_);
//            mapObsP2Pl[iReadNext][i] = pl.p2plDist(pl_w[ids[i]].p_);
//            mapObsNum[iReadNext][i] = 1;
          if (ids(i) != idsPrev(i)) {
            mapObsDot[iReadNext][i] = 0.;
            mapObsP2Pl[iReadNext][i] = 0.;
            mapObsNum[iReadNext][i] = 0.;
//            std::cout << "resetting " << iReadNext << " " << i << std::endl;
          }
          if (values(i) > 0.01) {
            ids(i) = -1;
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
        iReadNext = (iReadNext+1)%sizeToRead;
        {
          std::lock_guard<std::mutex> lock(nnLock); 
//          mapObsDot.iInsert_ = std::max(iReadNext;
//          mapObsP2Pl.iInsert_ = std::max(iReadNext;
          nn.iInsert_ = std::max(iReadNext, nn.iInsert_);
        }
        idNNUpdate = iReadNext;
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
  Eigen::Matrix3f InfoO = 10000.*Eigen::Matrix3f::Identity();
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
      // compute gradient
      tdp::VectorkNNida& ids = nn.GetCircular(i);
      tdp::Plane& pl = pl_w.GetCircular(i);
//      tdp::Vector3fda& Jn = Jn_w[i];
//      tdp::Vector3fda& Jp = Jp_w[i];
      tdp::Vector3fda Jn = tdp::Vector3fda::Zero();
      tdp::Vector3fda Jp = tdp::Vector3fda::Zero();
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
//          std::cout << Jn.transpose() << " " << pl.z_ <<
//            vmfs[pl.z_].mu_.transpose() << ", " << vmfs[pl.z_].tau_ << std::endl;
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
      bool haveFullNeighborhood = (ids.array() >= 0).all();
      if (haveFullNeighborhood) {
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
            }
          }
        }
      }
      tdp::Vector3fda mu;
        if (doVariationalUpdate && haveFullNeighborhood) {
          Eigen::Matrix3f SigmaPl;
          Eigen::Matrix3f Info =  InfoO*numSum_w[i];
  //        Eigen::Vector3f xi = SigmaO.ldlt().solve(pl.p_);
          Eigen::Vector3f xi = Info*pl.p_; //*pl.w_;
          for (int i=0; i<kNN; ++i) {
            if (ids[i] > -1) {
//              && zS[ids[i]] < Ksample ) {
//              SigmaPl = vmfs[zS[ids[i]]].mu_*vmfs[zS[ids[i]]].mu_.transpose();
              SigmaPl = pl_w[ids[i]].n_*pl_w[ids[i]].n_.transpose();
              Info += SigmaPl;
              xi += SigmaPl*pl_w[ids[i]].p_;
            }
          }
//          Eigen::Matrix3f Sigma = Info.inverse();
          mu = Info.ldlt().solve(xi);
  //        std::cout << xi.transpose() << " " << mu.transpose() << std::endl;
//          pi = Normal<float,3>(mu, Sigma).sample(rnd);
//          pl.p_ = mu;
        }
//      std::cout << "have " << numGrads << " gradients > 0  off " << iInsert-iRead  << std::endl;
      // apply gradient
//      tdp::Vector3fda& Jn = Jn_w[i];
//      tdp::Vector3fda& Jp = Jp_w[i];
      {
        std::lock_guard<std::mutex> mapGuard(mapLock);
        pl.n_ = (pl.n_- alphaGrad * Jn).normalized();
        if (haveFullNeighborhood &&  doVariationalUpdate) {
          pl.p_ = mu;
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

  std::vector<std::vector<uint32_t>> invInd;
  std::vector<size_t> id_w;
  id_w.reserve(MAP_SIZE);

//  std::random_device rd;
  std::mt19937 gen(19023);

  std::vector<uint32_t> idNew;
  idNew.reserve(w*h);

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
//    
//    if (useTexture.GuiChanged() && useTexture) {
//      HThr = -32;
//      negLogEvThr = -4;
//    }

    idNew.clear();
    if (!gui.paused() && !gui.finished()
        && frame > 0
        && (runMapping || frame == 1) 
        && (trackingGood || frame < 10)) { // add new observations

      // update mask only once to know where to insert new planes
      TICK("data assoc");
      projAssoc.Associate(vbo_w, nbo_w, tbo, T_wc.Inverse(), dMin,
          dMax, std::max(0, frame-dtAssoc), pl_w.SizeToRead());
      TOCK("data assoc");
      TICK("extract assoc");
//      z.Fill(0);
      idsCur.clear();
//      projAssoc.GetAssoc(z, mask, idsCur);
      projAssoc.GetAssocOcclusion(pl_w, pc, T_wc.Inverse(),
          occlusionDepthThr, z, mask, idsCur);
      std::random_shuffle(idsCur.begin(), idsCur.end());
      TOCK("extract assoc");

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
             mask, W, frame, T_wc, cam, dpc, pl_w, pc_w, pc0_w, rgb_w,
            n_w, grad_w, rs, ts);
        TOCK("normals");
        TICK("add to model");
        if (!runSampling.Get()) {
          for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
            dpvmf.addObservation(&pl_w[i].n_, &pl_w[i].z_);
          }
        }
        for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
          gradDir_w[i] = pl_w[i].grad_.normalized();
          pcSum_w[i] = pl_w[i].p_;
          nSum_w[i] = pl_w[i].n_;
          numSum_w[i] ++;
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

      TOCK("add to model");
      numMapPoints = pl_w.SizeToRead();
//      if (gui.verbose) 
//        std::cout << " # map points: " << pl_w.SizeToRead() 
//          << " " << dpvmf.GetZs().size() << std::endl;
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
            invInd.back().reserve(1000);
          } else {
            invInd[k].clear();
          }
        }
        if (pruneAssocByRender) {
          // only use ids that were found by projecting into the current pose
          for (auto i : idsCur) {
            uint32_t k = pl_w[i].z_;
            if (invInd[k].size() < 1000)
              invInd[k].push_back(i);
          }
        } else {      
          id_w.resize(pl_w.SizeToRead());
          std::iota(id_w.begin(), id_w.end(), 0);
          std::random_shuffle(id_w.begin(), id_w.end());
          // use all ids in the current map
          for (auto i : id_w) {
            uint32_t k = pl_w[i].z_;
            if (invInd[k].size() < 1000)
              invInd[k].push_back(i);
          }
        }
//        uint32_t Kcur = K;
      }
      TOCK("dpvmf");
    }

//    glClearColor(bgGrey, bgGrey, bgGrey, 1.0f);
//    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

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
    pyrGreyFl.CopyFrom(cuPyrGreyFlSmooth);
    greyFl = pyrGreyFl.GetImage(0);

    cuGradGrey = cuPyrGradGrey.GetImage(0);
    tdp::Gradient(cuGreyFlSmooth, cuGreyDu, cuGreyDv, cuGradGrey);
    if (smoothGreyPyr==1) {
      tdp::CompletePyramidBlur9(cuPyrGradGrey, 1.);
    } else {
      tdp::CompletePyramidBlur(cuPyrGradGrey, 1.);
    }
    pyrGradGrey.CopyFrom(cuPyrGradGrey);
    gradGrey = pyrGradGrey.GetImage(0);

    tdp::Gradient2AngleNorm(cuGreyDu, cuGreyDv, cuGreyGradTheta,
        cuGreyGradNorm);
    greyGradNorm.CopyFrom(cuGreyGradNorm);

    n.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    trackingGood = false;
    if (frame > 1 && runTracking && !gui.finished()) { // tracking
      mask.Fill(0);
      if (doSO3prealign) {
        if (gui.verbose) std::cout << "SO3 prealignment" << std::endl;
        TICK("icp RGB");
        Eigen::Matrix<float,3,3> A;
        Eigen::Matrix<float,3,1> b;
        Eigen::Matrix<float,3,1> Ai;
        for (int32_t pyr=SO3maxLvl; pyr>=0; --pyr) {
          float scale = pow(0.5,pyr);
          CameraT camLvl = cam.Scale(scale);
          tdp::Image<float> greyFlLvl = pyrGreyFl.GetImage(pyr);
          tdp::Image<tdp::Vector2fda> gradGreyLvl = pyrGradGrey.GetImage(pyr);
          if (gui.verbose) std::cout << "pyramid lvl " << pyr << " scale " << scale << std::endl;
          for (size_t it = 0; it < SO3maxIt*(pyr+1); ++it) {
//            for (auto& ass : assoc) mask[ass.second] = 0;
//            assoc.clear();
            A = Eigen::Matrix<float,3,3>::Zero();
            b = Eigen::Matrix<float,3,1>::Zero();
            Ai = Eigen::Matrix<float,3,1>::Zero();
            float err = 0.;
            float H = 1e10;
            float Hprev = 1e10;
            tdp::SE3f T_cw = T_wc.Inverse();
            for (auto& i : idsCur) {
              tdp::Plane& pl = pl_w.GetCircular(i);
              Eigen::Vector2f x = camLvl.Project(T_cw*pl.p_);
              float u = x(0);
              float v = x(1);
              if (0 > u || u >= w*scale || 0 > v || v >= h*scale) 
                continue;
              if (!AccumulateIntDiff(pl, T_cw, camLvl, greyFlLvl.GetBilinear(u,v),
                    gradGreyLvl.GetBilinear(u,v), lambdaTex, A, Ai, b, err))
                continue;
//              mask(u,v) |= 1;
//              assoc.emplace_back(i,u+v*w);
              //tdp::CheckEntropyTermination(A, Hprev, SO3HThr,
              //    SO3condEntropyThr, SO3negLogEvThr, H, gui.verbose);
              //  break;
              //Hprev = H;
            }
            Eigen::Matrix<float,3,1> x = Eigen::Matrix<float,3,1>::Zero();
            if (assoc.size() > 10) {
              // solve for x using ldlt
              x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
              T_wc.rotation() = T_wc.rotation() * tdp::SO3f::Exp_(x);
            }
            bool term = (x.norm()*180./M_PI < icpdRThr
                && tdp::CheckEntropyTermination(A, Hprev, SO3HThr, 0.f,
                  SO3negLogEvThr, H, gui.verbose));
            if (gui.verbose) {
              std::cout << "\tit " << it << ": err=" << err 
                << "\tH: " << H 
                << "\t# inliers: " << assoc.size()
                << "\t|x|: " << x.norm()*180./M_PI << std::endl;
            }
            if (term) break;
          }
        }
        TOCK("icp RGB");
      }
      if (runICP) {
        if (gui.verbose) std::cout << "SE3 ICP" << std::endl;
        TICK("icp");
        std::vector<size_t> indK(invInd.size(),0);
        Eigen::Matrix<float,6,6> A;
        Eigen::Matrix<float,6,1> b;
        Eigen::Matrix<float,6,1> Ai;
        float dotThr = cos(angleThr*M_PI/180.);
        for (size_t it = 0; it < maxIt; ++it) {
          for (auto& ass : assoc) mask[ass.second] = 0;
          assoc.clear();
          indK = std::vector<size_t>(invInd.size(),0);
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
            k = (k+1) % invInd.size();
            while (indK[k] < invInd[k].size()) {
              size_t i = invInd[k][indK[k]++];
              int32_t u, v;
              tdp::Plane& pl = pl_w.GetCircular(i);
              numProjected = numProjected + 1;

              if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, cam, pc,
                    W, dpc, n, curv, u,v ))
                continue;
              if (useTexture) {
                if (!AccumulateP2PlIntensity(pl, T_wc, T_cw, cam, pc(u,v),
                      n(u,v), greyFl(u,v), gradGrey(u,v), distThr, p2plThr, dotThr,
                      lambdaTex, A, Ai, b, err))
                  continue;
              } else if (useNormals) {
                if (!AccumulateP2PlNormal(pl, T_wc, T_cw, cam, pc(u,v),
                      n(u,v), distThr, p2plThr, dotThr, lambdaNsOld, A,
                      Ai, b, err)) {
                  continue;
                }
              } else if (useNormalsAndTexture) {
                if (!AccumulateP2PlIntensityNormals(pl, T_wc, T_cw, cam, pc(u,v),
                      n(u,v), greyFl(u,v),gradGrey(u,v), distThr, p2plThr, dotThr,
                      lambdaNs, lambdaTex, A, Ai, b, err)) {
                  continue;
                }
              } else {
                if (!AccumulateP2Pl(pl, T_wc, T_cw, pc(u,v), n(u,v),
                      distThr, p2plThr, dotThr, A, Ai, b, err))
                  continue;
              }
              pl.lastFrame_ = frame;
              ts[i] = frame;
              pl.numObs_ ++;
              mask(u,v) |= 1;
              assoc.emplace_back(i,u+v*pc.w_);
              break;
            }

            if (k == 0) {
              if (tdp::CheckEntropyTermination(A, Hprev, HThr, condEntropyThr, 
                    negLogEvThr, H, gui.verbose))
                break;
              Hprev = H;
            }
            exploredAll = true;
            for (size_t k=0; k<indK.size(); ++k) exploredAll &= indK[k] >= invInd[k].size();
          }
          numInl = assoc.size();
          Eigen::Matrix<float,6,1> x = Eigen::Matrix<float,6,1>::Zero();
          if (assoc.size() > 6) { // solve for x using ldlt
//            std::cout << "A: " << std::endl << A << std::endl << "b: " << b.transpose() << std::endl;
            x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
            T_wc = T_wc * tdp::SE3f::Exp_(x);
          }
          if (gui.verbose) {
            std::cout << "\tit " << it << ": err=" << err 
              << "\t# inliers: " << numInl
              << "\t|x|: " << x.topRows(3).norm()*180./M_PI 
              << " " <<  x.bottomRows(3).norm() << std::endl;
          }
          if (x.topRows<3>().norm()*180./M_PI < icpdRThr
              && x.bottomRows<3>().norm() < icpdtThr
              && tdp::CheckEntropyTermination(A, Hprev, HThr, 0.f,
                negLogEvThr, H, gui.verbose)) {
            break;
          }
        }

        if (gui.verbose) {
          for (size_t k=0; k<invInd.size(); ++k) {
            if (invInd[k].size() > 0 )
              std::cout << "used different directions " << k << "/" 
                << invInd.size() << ": " << indK[k] 
                << " of " << invInd[k].size() << std::endl;
          }
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
        trackingGood = H <= HThr && assoc.size() > 10;
        TOCK("icp");
        if (trackingGood) if (gui.verbose) std::cout << "tracking good" << std::endl;
      }

      if (updatePlanes && trackingGood) {
        std::lock_guard<std::mutex> mapGuard(mapLock);
        TICK("update planes");
        size_t numNN = 0;
//        tdp::SE3f T_cw = T_wc.Inverse();
        for (const auto& ass : assoc) {

          int32_t u = ass.second%pc.w_;
          int32_t v = ass.second/pc.w_;

          tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);
          tdp::Vector3fda n_c_in_w = T_wc.rotation()*n(u,v);

          float w = numSum_w[ass.first];
          // filtering grad grey
          pl_w[ass.first].grad_ = (pl_w[ass.first].grad_*w 
              + pl_w[ass.first].Compute3DGradient(T_wc, cam, u, v, gradGrey(u,v)))/(w+1);
          pl_w[ass.first].grey_ = (pl_w[ass.first].grey_*w + greyFl(u,v)) / (w+1);
          pl_w[ass.first].rgb_ = ((pl_w[ass.first].rgb_.cast<float>()*w
                + rgb(u,v).cast<float>()) / (w+1)).cast<uint8_t>();

          grad_w[ass.first] = pl_w[ass.first].grad_;
          gradDir_w[ass.first] = grad_w[ass.first].normalized();
          pcSum_w[ass.first] += pc_c_in_w;
          nSum_w[ass.first] += n_c_in_w;
          numSum_w[ass.first] ++;

          if (updateMap) {
            for (size_t i=0; i<kNN; ++ i) {
              for (const auto& assB : assoc) {
                if (assB.first == nn[ass.first](i)){
                  int32_t uB = assB.second%pc.w_;
                  int32_t vB = assB.second/pc.w_;
                  //mapObsP2Pl[ass.first](i) += n(u,v).dot(pc(uB,vB)-pc(u,v));
                  //mapObsDot[ass.first](i) += n(u,v).dot(n(uB,vB));
                  //mapObsNum[ass.first](i) ++;
                  float w = mapObsNum[ass.first](i);
                  mapObsP2Pl[ass.first](i) = (w*mapObsP2Pl[ass.first](i) + n(u,v).dot(pc(uB,vB)-pc(u,v)))/(w+1.);
                  mapObsDot[ass.first](i) = (mapObsDot[ass.first](i)*w + n(u,v).dot(n(uB,vB)))/(w+1.);
                  mapObsNum[ass.first](i) = std::min(100.f,w+1.f) ;
                  //std::cout << "found NN " << i << " of " << ass.first 
                  //  << " " << mapObsP2Pl[ass.first](i)/mapObsNum[ass.first](i) 
                  //  << " " << mapObsDot[ass.first](i)/mapObsNum[ass.first](i)
                  //  << " " << mapObsNum[ass.first](i) << std::endl;
                  numNN++;
                  break;
                }
              }
            }
          } else {
            pl_w[ass.first].AddObs(pc_c_in_w, n_c_in_w);
            n_w[ass.first] =  pl_w[ass.first].n_;
            pc_w[ass.first] = pl_w[ass.first].p_;
          }
        }
        if (gui.verbose) std::cout << "num NN measured " << numNN << std::endl;
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
      TICK("Draw 3D vbo cbo upload");
      vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
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
        if (showAge || showObs || showCurv || showGrey) {
          if (showAge) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = ts.GetCircular(i);
          } else if (showObs) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).numObs_;
          } else if (showGrey) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).grey_;
          } else {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).curvature_;
          }
          valuebo.Upload(age.ptr_, pl_w.SizeToRead()*sizeof(float), 0);
          std::pair<float,float> minMaxAge = age.GetRoi(0,0,
              pl_w.SizeToRead(),1).MinMax();
          tdp::RenderVboValuebo(vbo_w, valuebo, minMaxAge.first,
              minMaxAge.second, P, MV);
        } else if (showLabels && frame > 1) {
          if (showDPvMFlabels) {
            lbo.Upload(zS.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
          } else {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              labels[i] = pl_w.GetCircular(i).z_;
            lbo.Upload(labels.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
          }
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
          glColor4f(0.3,0.3,0.3,0.3);
          for (auto& ass : mapNN) {
            if (ass.second >= 0)
              tdp::glDrawLine(pl_w[ass.first].p_, pl_w[ass.second].p_);
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
      if (frame > 1) {
        if (showDPvMFlabels) {
          lbo.Upload(zS.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
        tdp::RenderLabeledVbo(nbo_w, lbo, normalsCam);
        } else if (showLabels) {
          for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
            labels[i] = pl_w.GetCircular(i).z_;
          lbo.Upload(labels.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
        tdp::RenderLabeledVbo(nbo_w, lbo, normalsCam);
        } else {
          glColor4f(0,0,1,0.5);
          pangolin::RenderVbo(nbo_w);
        }
      }  
      if (!runSampling.Get()) {
        glColor4f(1,0,0,1.);
        for (size_t k=0; k<dpvmf.GetK(); ++k) {
          tdp::glDrawLine(tdp::Vector3fda::Zero(), dpvmf.GetCenter(k));
        }
      } else { 
        glColor4f(0,1,0,1.);
        {
          std::lock_guard<std::mutex> lock(vmfsLock);
          for (size_t k=0; k<vmfs.size(); ++k) {
            if (vmfSS[k](3) > 0)
              tdp::glDrawLine(tdp::Vector3fda::Zero(), vmfs[k].mu_);
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

//  imuInterp.Stop();
//  if (imu) imu->Stop();
//  delete imu;
//  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

