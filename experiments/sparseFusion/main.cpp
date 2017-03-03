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
#define zKTrac 3
#define kNN 12
#define MAP_SIZE 1000000

namespace tdp {

typedef Eigen::Matrix<float,  zKTrac,1,Eigen::DontAlign> VectorZfda;
typedef Eigen::Matrix<uint16_t,zKTrac,1,Eigen::DontAlign> VectorZuda;

void AccumulateGaussianSS(
    const tdp::Vector3fda& x,
    tdp::Vector3fda& xSum,
    tdp::Matrix3fda& xOuter,
    float& xCount
    ) {
  xSum += x; 
  xOuter += x*x.transpose(); 
  xCount ++;
}

void RunningAvgGaussianSS(
    const tdp::Vector3fda& x,
    float xCountMax,
    tdp::Vector3fda& xSum,
    tdp::Matrix3fda& xOuter,
    float& xCount
    ) {
  xSum = (xSum*xCount + x)/(xCount+1); 
  xOuter = (xOuter*xCount + x*x.transpose())/(xCount+1);
  xCount = std::min(xCountMax, xCount+1);
}

void RunningAvgvMFSS(
    const tdp::Vector3fda& x,
    float xCountMax,
    tdp::Vector3fda& xSum,
    float& xCount
    ) {
  xSum = (xSum*xCount + x)/(xCount+1); 
  xCount = std::min(xCountMax, xCount+1);
}

void InsertLabelML(VectorZuda& ids, VectorZfda& counts, uint16_t id,
    uint16_t& idMax, float& countMax) {
  int32_t kMatch=-1;
  int32_t kMin=-1;
  float countMin=std::numeric_limits<float>::max();
  int32_t kMax=-1;
  countMax=std::numeric_limits<float>::lowest();
  for (int32_t k=0; k<zKTrac; ++k) {
    if (ids(k) == id) {
      kMatch = k;
    }
    if (counts(k) < countMin) {
      countMin = counts(k);
      kMin = k;
    }
    if (counts(k) > countMax) {
      countMax = counts(k);
      idMax = ids(k);
    }
  }
  if (kMatch >= 0) {
    counts(kMatch) ++;
    countMax = idMax == kMatch? countMax+1 : countMax;
  } else {
    counts(kMin) = 1;
    ids(kMin) = id;
  }
}

typedef Eigen::Matrix<float,kNN,1,Eigen::DontAlign> VectorkNNfda;
typedef Eigen::Matrix<int32_t,kNN,1,Eigen::DontAlign> VectorkNNida;

void InflateObsCovByTransformationCov(
    const tdp::SE3f& T_wc, 
    const tdp::Vector3fda& p_w,
    const Eigen::Matrix<float,6,6>& Sigma_wc,
    tdp::Matrix3fda& SigmaO
    ) {
  Eigen::Matrix<float, 6, 3> Jse3;
  Jse3 << tdp::SO3f::invVee( T_wc.rotation().matrix().transpose()*(p_w-T_wc.translation())), -Eigen::Matrix3f::Identity();

//  std::cout << "before" << std::endl << SigmaO << std::endl;
//  std::cout << "Sigma_wc" << std::endl << Sigma_wc << std::endl;
//  std::cout << "Jse3" << std::endl << Jse3 << std::endl;
  SigmaO += Jse3.transpose()*Sigma_wc*Jse3;
//  std::cout << "after" << std::endl << SigmaO << std::endl;
}

//void InflateObsTauByTransformationCov(
//    const tdp::SE3f& T_wc, 
//    const tdp::Vector3fda& p_w,
//    const Eigen::Matrix<float,6,6>& Sigma_wc,
//    tdp::Matrix3fda& SigmaO
//    ) {
//  Eigen::Matrix<float, 6, 3> Jse3;
//  Jse3 << tdp::SO3f::invVee( T_wc.rotation().matrix().transpose()*(p_w-T_wc.translation())), -Eigen::Matrix3f::Identity();
//  SigmaO += Jse3*Sigma_wc*Jse3.transpose();
//}

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
    const Eigen::Matrix<float,6,6>& Sigma_wc,
    const CameraT& cam,
    const Image<float>& rho, 
    const Image<Vector3fda>& rays, 
    const Image<Vector6dda>& outerRaysInt, 
    Image<Vector4fda>& dpc, 
    ManagedHostCircularBuffer<Plane>& pl_w,
    ManagedHostCircularBuffer<Vector3fda>& pc_w,
    ManagedHostCircularBuffer<Matrix3fda>& pc0Info_w,
    ManagedHostCircularBuffer<Matrix3fda>& pc0Cov_w,
    ManagedHostCircularBuffer<Vector3bda>& rgb_w,
    ManagedHostCircularBuffer<Vector3fda>& n_w,
    ManagedHostCircularBuffer<Vector3fda>& grad_w,
    ManagedHostCircularBuffer<float>& ImSum,
    ManagedHostCircularBuffer<float>& ImSqSum,
    ManagedHostCircularBuffer<float>& ImCount,
    ManagedHostCircularBuffer<float>& ImVar,
    ManagedHostCircularBuffer<float>& rs,
    ManagedHostCircularBuffer<uint16_t>& ts,
    int normalMethod,
    bool useTrackingUncertainty
    ) {
  Plane pl;
  tdp::Brief feat;
  Vector3fda n, p;
  float curv = 0;
  float radiusStd = 0.;
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
      if (normalMethod == 0) {
//        success = tdp::normalExtractMethod(pc, u, v, Wscaled, 0.29, dpc, n, curv, p);
        success = tdp::NormalViaScatterAproxIntInvD(rho, rays,
            outerRaysInt, u, v, Wscaled, n);
        p = pc(u,v);
      } else if (normalMethod == 1) {
        success = tdp::NormalViaScatterUnconstrained(pc, u, v, Wscaled,
            n, curv, radiusStd);
        p = pc(u,v);
      } else {
        success = tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, dpc, n, curv, radiusStd, p);
      }
      if (success) {
//        std::cout << "extracted normal at " << u << "," << v << std::endl;
//        ExtractClosestBrief(pc, grey, pts, orientation, 
//            p, n, T_wc, cam, Wscaled, i%mask.w_, i/mask.w_, feat);
        pl.p_ = T_wc*p;
        pl.n_ = T_wc.rotation()*n;
        pl.p0_ = pl.p_;
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
//        pl.feat_ = feat;
//        pl.r_ = 2*W*pc[i](2)/cam.params_(0); // unprojected radius in m
        if (normalMethod < 2) {
          //http://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00677.pdf
          // radius covering a single pizel
          pl.r_ = p(2)/(1.41421*cam.params_(0)*n(2)); // unprojected radius in m
        } else {
          pl.r_ = radiusStd*3;
        }

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
        if (useTrackingUncertainty) {
          tdp::InflateObsCovByTransformationCov(T_wc, pl.p_, Sigma_wc, SigmaO);
        }
        SigmaO = T_wc.rotation().matrix()*SigmaO*T_wc.rotation().matrix().transpose();

        pl.Hp_ = 0.5*SigmaO.eigenvalues().array().real().log().sum();

        pc0Cov_w.Insert(SigmaO);
        pc0Info_w.Insert(SigmaO.inverse());
        pl_w.Insert(pl);
        pc_w.Insert(pl.p_);
        n_w.Insert(pl.n_);
        grad_w.Insert(pl.grad_);
        rgb_w.Insert(pl.rgb_);
        rs.Insert(pl.r_);
        ts.Insert(pl.lastFrame_);

        ImSum.Insert(greyFl[i]);
        ImSqSum.Insert(greyFl[i]*greyFl[i]);
        ImCount.Insert(1);
        ImVar.Insert(1.); 
      }
    }
  }
    }
}

bool ProjectiveAssocOcl(const tdp::Vector3fda& p_w,
    tdp::SE3f& T_wc, 
    CameraT& cam,
    const Image<float>& d,
    float occlusionDepthThr,
    int32_t& u,
    int32_t& v
    ) {
  tdp::Vector3fda pc_w_in_c = T_wc.Inverse()*p_w;
  Eigen::Vector2f x = cam.Project(pc_w_in_c);
  if (!d.Inside(x)) 
    return false;
  u = floor(x(0)+0.5f);
  v = floor(x(1)+0.5f);
  float d_c = d(u,v);
  if (d_c != d_c || fabs(d_c-pc_w_in_c(2)) > occlusionDepthThr) 
    return false;
  return true;
}

bool ProjectiveAssocOcl(const Plane& pl, 
    tdp::SE3f& T_wc, 
    CameraT& cam,
    const Image<float>& d,
    float occlusionDepthThr,
    int32_t& u,
    int32_t& v
    ) {
  if (!pl.valid_)
    return false;
  return ProjectiveAssocOcl(pl.p_, T_wc, cam, d, occlusionDepthThr, u,v);
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
    const Image<float>& rho, 
    const Image<Vector3fda>& rays, 
    const Image<Vector6dda>& outerRaysInt, 
    uint32_t W,
    Image<Vector3fda>& n,
    Image<float>& curv,
    Image<float>& rad,
    int32_t u,
    int32_t v,
    int normalMethod
    ) {
  if (0 <= u && u < pc.w_ && 0 <= v && v < pc.h_) {
    if (tdp::IsValidData(pc(u,v))) {
//      uint32_t Wscaled = floor(W*pc(u,v)(2));
      uint32_t Wscaled = W;
      tdp::Vector3fda ni = n(u,v);
      tdp::Vector3fda pi = pc(u,v);
      float curvi=0.;
      float radiusi=0.;
      if (!tdp::IsValidData(ni)) {
//        if(tdp::NormalViaScatter(pc, u, v, Wscaled, ni)) {
//        if(tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, dpc, ni, curvi, pi)) {
        bool success = false;
        if (normalMethod == 0) {
//          success = tdp::normalExtractMethod(pc, u, v, Wscaled, 0.29, 
//              dpc, ni, curvi, pi);
//          //http://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00677.pdf
//          // radius covering a single pizel
//          // TODO: 550 -> fu
          success = tdp::NormalViaScatterAproxIntInvD(rho, rays,
              outerRaysInt, u, v, Wscaled, ni);
          radiusi = pi(2)/(1.41421*550.*ni(2)); // unprojected radius in m
        } else if (normalMethod == 1) {
          success = tdp::NormalViaScatterUnconstrained(pc, u, v,
              Wscaled, ni, curvi, radiusi);
          radiusi = pi(2)/(1.41421*550.*ni(2)); // unprojected radius in m
        } else {
          success = tdp::NormalViaVoting(pc, u, v, Wscaled, 0.29, 
              dpc, ni, curvi, radiusi, pi);
          radiusi *= 3.;
        }
        if (success) {
          n(u,v) = ni;
          pc(u,v) = pi;
          curv(u,v) = curvi;
          rad(u,v) = radiusi;
          return true;
        }
      } else {
        return true;
      }
    }
  }
  return false;
}

//bool ProjectiveAssocNormalExtract(const Plane& pl, 
//    tdp::SE3f& T_cw, 
//    CameraT& cam,
//    Image<Vector3fda>& pc,
//    uint32_t W,
//    Image<Vector4fda>& dpc,
//    Image<Vector3fda>& n,
//    Image<float>& curv,
//    int32_t& u,
//    int32_t& v,
//    bool normalExtractMethod
//    ) {
//  const tdp::Vector3fda& n_w =  pl.n_;
//  const tdp::Vector3fda& pc_w = pl.p_;
//  Eigen::Vector2f x = cam.Project(T_cw*pc_w);
//  u = floor(x(0)+0.5f);
//  v = floor(x(1)+0.5f);
//  return EnsureNormal(pc, dpc, W, n, curv, u, v, normalExtractMethod);
//}


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
    float sqrtInfoP2Pl,
    float sqrtInfoIm,
    Eigen::Matrix<float,6,6>& Ap2pl,
    Eigen::Matrix<float,6,1>& bp2pl,
    Eigen::Matrix<float,6,6>& Aphoto,
    Eigen::Matrix<float,6,1>& bphoto,
    Eigen::Matrix<float,6,1>& Ai,
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
        Ap2pl +=   sqrtInfoP2Pl*(Ai * Ai.transpose());
        bp2pl +=   sqrtInfoP2Pl*(Ai * bi);
        err += sqrtInfoP2Pl*(bi);
//        std::cout << " p2pl " << bi << " " << Ai.transpose() << std::endl;
        // texture inverse transform verified Jse3 
        Eigen::Matrix<float,2,3> Jpi = cam.Jproject(T_cw*pc_w);
        Eigen::Matrix<float,3,6> Jse3;
        Jse3 << SO3mat<float>::invVee(T_cw.rotation()*(pc_w-T_wc.translation())), 
             -Eigen::Matrix3f::Identity();
        Ai = Jse3.transpose() * Jpi.transpose() * gradGrey_ci;
        bi = - grey_ci + pl.grey_;
        Aphoto += sqrtInfoIm*(Ai * Ai.transpose());
        bphoto += sqrtInfoIm*(Ai * bi);

//        std::cout << " intensity " << bi << " " << Ai.transpose() << std::endl;
//        std::cout << Jse3 << std::endl;
//        std::cout << Jpi << std::endl;
//        std::cout << gradGrey_ci << std::endl;
        err += sqrtInfoIm*bi;
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
  std::string varsMapFile = "";
  std::string varsIcpFile = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
    varsMapFile = (argc > 3) ? std::string(argv[3]) : "";
    varsIcpFile = (argc > 4) ? std::string(argv[4]) : "";
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
  pangolin::CreatePanel("visPanel").SetBounds(0.3,1.,pangolin::Attach::Pix(180),pangolin::Attach::Pix(360));
  pangolin::Display("visPanel").Show(false);
  pangolin::CreatePanel("mapPanel").SetBounds(0.3,1.,pangolin::Attach::Pix(360),pangolin::Attach::Pix(540));
  pangolin::Display("mapPanel").Show(false);
  pangolin::CreatePanel("icpPanel").SetBounds(0.3,1.,pangolin::Attach::Pix(360),pangolin::Attach::Pix(540));
  pangolin::Display("icpPanel").Show(false);

  gui.container().SetLayout(pangolin::LayoutEqual);

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,true);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams,1);
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
  viewGrads.Show(false);

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
  viewGreyGradNorm.Show(false);
  plotters.Show(false);

  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<float> curv(wc,hc);
  tdp::ManagedHostImage<float> rad(wc,hc);
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
  tdp::ManagedHostPyramid<tdp::Vector6fda,PYR> pyrOuterRays(wc,hc);
  tdp::ManagedHostPyramid<tdp::Vector6dda,PYR> pyrOuterRaysInt(wc+(1<<PYR),hc+(1<<PYR));
  tdp::Image<tdp::Vector3fda> rays;
  tdp::Image<tdp::Vector6fda> outerRays;
  tdp::Image<tdp::Vector6dda> outerRaysInt;
  // precompute outer products of rays for surface normal extraction
  for (size_t lvl=0; lvl<PYR; ++lvl) {
    rays = pyrRay.GetImage(lvl);
    outerRays = pyrOuterRays.GetImage(lvl);
    outerRaysInt = pyrOuterRaysInt.GetImage(lvl);
    for (size_t i=0; i<outerRays.Area(); ++i) {
      outerRays[i](0) = rays[i](0)*rays[i](0);
      outerRays[i](1) = rays[i](0)*rays[i](1);
      outerRays[i](2) = rays[i](0)*rays[i](2);
      outerRays[i](3) = rays[i](1)*rays[i](1);
      outerRays[i](4) = rays[i](1)*rays[i](2);
      outerRays[i](5) = rays[i](2)*rays[i](2);
    }
    // construct integral image of outer products of rays for surface normal extraction
    outerRaysInt.Fill(tdp::Vector6dda::Zero());
    for (size_t u=1; u<rays.w_+1; ++u) {
      for (size_t v=1; v<rays.h_+1; ++v) {
        outerRaysInt(u,v) = -outerRaysInt(u-1,v-1) 
          + outerRaysInt(u-1,v)+outerRaysInt(u,v-1)+outerRays(u,v).cast<double>();
      }
    }
  }
  rays = pyrRay.GetImage(0);
  outerRays = pyrOuterRays.GetImage(0);
  outerRaysInt = pyrOuterRaysInt.GetImage(0);

  // ICP stuff
  tdp::ManagedDevicePyramid<tdp::Vector3fda,PYR> cuPyrPc(wc,hc);
  tdp::Image<tdp::Vector3fda> cuPc = cuPyrPc.GetImage(0);
  tdp::ManagedHostPyramid<tdp::Vector3fda,PYR> pyrPc(wc,hc);
  tdp::Image<tdp::Vector3fda> pc = pyrPc.GetImage(0);
  pc.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  tdp::ManagedHostPyramid<tdp::Vector3fda,PYR> pyrN(wc,hc);
  tdp::Image<tdp::Vector3fda> n = pyrN.GetImage(0);

  tdp::ManagedDevicePyramid<float,PYR> cuPyrRho(wc,hc);
  tdp::ManagedHostPyramid<float,PYR> pyrRho(wc,hc);
  tdp::Image<float> rho = pyrRho.GetImage(0);
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

  pangolin::Var<bool> record("ui.record",false,false);
  pangolin::Var<bool> snapShot("ui.snapshot",false,false);
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,10.);
  pangolin::Var<bool> showVisPanel("ui.viz panel",false,true);
  pangolin::Var<bool> showIcpPanel("ui.icp panel",false,true);
  pangolin::Var<bool> showMapPanel("ui.map panel",false,true);

  pangolin::Var<bool> savePly("ui.save ply",false,true);

  pangolin::Var<int> numMapPoints("ui.num Map",0,0,0);
  pangolin::Var<int> numPruned("ui.num Pruned",0,0,0);
  pangolin::Var<int> numProjected("ui.num Proj",0,0,0);
  pangolin::Var<int> numInl("ui.num Inl",0,0,0);
  pangolin::Var<int> idMapUpdate("ui.id Map",0,0,0);
  pangolin::Var<int> idNNUpdate("ui.id NN",0,0,0);
  pangolin::Var<int> idNNStack("ui.id NN Stack",0,0,0);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> runLoopClosureGeom("ui.run loop closure geom",false,true);

  pangolin::Var<bool> pruneNoise("ui.prune Noise",true,true);
  pangolin::Var<float> relPruneDistThr("ui.rel prune dist Thr",3.,1.,10);
  pangolin::Var<float> pruneHThr("ui.prune H Thr",-15,-40.,-20.);
  pangolin::Var<float> pruneNumObsThr("ui.prune NumObsThr",10,3,30);
  pangolin::Var<int> survivalTime("ui.survival Time",100,0,200);
  pangolin::Var<int> minNumObs("ui.min Obs",10,1,20);
  pangolin::Var<int> numAdditionalObs("ui.num add Obs",3000,0,1000);

  pangolin::Var<int> smoothGrey("ui.smooth grey",1,0,2);
  pangolin::Var<int> smoothGreyPyr("ui.smooth grey pyr",1,0,1);
  pangolin::Var<int> smoothDPyr("ui.smooth D pyr",1,0,1);
  pangolin::Var<int> normalExtractMethod("ui.normal Extr Method",1,0,2);
  pangolin::Var<int>  W("ui.W ",4,1,15);
  pangolin::Var<float> subsample("ui.subsample %",1.,0.1,3.);
  pangolin::Var<float> pUniform("ui.p uniform ",0.1,0.1,1.);

  pangolin::Var<bool> runMapping("mapPanel.run mapping",true,true);
  pangolin::Var<bool> updateMap("mapPanel.update map",true,true);
  pangolin::Var<bool> useTrackingUncertainty("mapPanel.use tracking uncertainty",true,true);
  pangolin::Var<bool> allowNNRevisit("mapPanel.revisit NNs",true, true);
  // TODO if sample normals if off then doRegvMF shoudl be on
//  pangolin::Var<bool> sampleNormals("mapPanel.sampleNormals",true,true);
  pangolin::Var<bool> useSurfNormalObs("mapPanel.use SurfNormal Obs",true,true);
  pangolin::Var<bool> normalsP2PlContrib("mapPanel.ns P2Pl contrib",true,true);
  pangolin::Var<bool> samplePoints("mapPanel.samplePoints",true,true);
  pangolin::Var<float> condHThr("mapPanel.condHThr",0.01,0.001,0.1);
  pangolin::Var<float> sampleCountThr("mapPanel.count Thr",5.,1.,100.);
  pangolin::Var<float> lambdaMRF("mapPanel.lamb z MRF",.1,0.01,10.);
  pangolin::Var<float> tauO("mapPanel.tauO",100.,0.0,200.);
  pangolin::Var<bool> estimateTauO("mapPanel.estimate TauO",false,true);
  pangolin::Var<bool> useSigmaPl("mapPanel.use sigmaPl",true,true);
  pangolin::Var<bool> useOtherNi("mapPanel.use OtherNi",false,true);
  pangolin::Var<float> sigmaPl("mapPanel.sigmaPl",0.01,0.01,.2);
  pangolin::Var<bool> estSigmaPl("mapPanel.est SigmaPl",false,true);
  pangolin::Var<bool> estSigmaIm("mapPanel.est SigmaIm",false,true);
  pangolin::Var<float> obsStdInflation("mapPanel.obsSigmaInfl",1,1,100);
  pangolin::Var<float> maxNnDist("mapPanel.max NN Dist",0.2, 0.1, 1.);
  pangolin::Var<float> alphaSchedule("mapPanel.alpha Schedule",10., 0.001, 1.);
  pangolin::Var<bool> sampleScheduling("mapPanel.sample scheduling",false,true);

  pangolin::Var<bool> runICP("icpPanel.run ICP",true,true);
  pangolin::Var<bool> icpReset("icpPanel.reset icp",true,false);
  pangolin::Var<int> maxIt0("icpPanel.max iter 0",5, 1, 20);
  pangolin::Var<int> maxIt1("icpPanel.max iter 1",7, 1, 20);
  pangolin::Var<int> maxIt2("icpPanel.max iter 2",10, 1, 20);
  pangolin::Var<int> maxIt3("icpPanel.max iter 3",15, 1, 20);
  pangolin::Var<int> ICPmaxLvl("icpPanel.icp max lvl",0, 0, PYR-1);

  pangolin::Var<bool> pruneAssocByRender("icpPanel.prune assoc by render",true,true);
  pangolin::Var<bool> semanticObsSelect("icpPanel.semObsSelect",true,true);
  pangolin::Var<bool> sortByGradient("icpPanel.sortByGradient",true,true);

  pangolin::Var<int> dtAssoc("icpPanel.dtAssoc",5000,1,1000);
  pangolin::Var<float> lambdaNs("icpPanel.lamb Ns",0.1,0.001,1.);
  pangolin::Var<float> lambdaTex("icpPanel.lamb Tex",0.1,0.0001,0.1);
  pangolin::Var<float> lambdaP2Pl("icpPanel.lamb p2pl",1.0,0.001,1.);
  pangolin::Var<bool> useTexture("icpPanel.use Tex ICP",true,true);
  pangolin::Var<bool> use3dGrads("icpPanel.use 3D grads ",false,true);
  pangolin::Var<bool> useNormals("icpPanel.use Ns ICP",false,true);
  pangolin::Var<bool> useNormalsAndTexture("icpPanel.use Tex&Ns ICP",false,true);
  pangolin::Var<bool> usevMFmeans("icpPanel.use vMF means",false,true);

  pangolin::Var<float> occlusionDepthThr("icpPanel.occlusion D Thr",0.06,0.01,0.3);
  pangolin::Var<float> numSigmaOclusion("icpPanel.num sigma ocl",30.,1.,6.);
  pangolin::Var<bool> sigmaOclusion("icpPanel.use sigma in ocl",true,true);
  pangolin::Var<float> angleThr("icpPanel.angle Thr",15, -1, 90);
  pangolin::Var<float> p2plThr("icpPanel.p2pl Thr",0.03,0,0.3);
  pangolin::Var<float> HThr("icpPanel.H Thr",-42.,-40.,-12.);
  pangolin::Var<float> negLogEvThr("icpPanel.neg log ev Thr",-9.5,-12.,-1.);
  pangolin::Var<float> dPyrHThr("icpPanel.d Pyr H Thr",4.,0.,8.);
  pangolin::Var<float> dPyrNewLogEvHThr("icpPanel.d Pyr H Thr",1.,0.,3.);
  pangolin::Var<float> dPyrdAlpha("icpPanel.d Pyr dAlpha",0.9,0.1,1.);
  pangolin::Var<float> condEntropyThr("icpPanel.rel log dH ", 1.e-3,1.e-3,1e-2);
  pangolin::Var<float> icpdRThr("icpPanel.dR Thr",0.25,0.1,1.);
  pangolin::Var<float> icpdtThr("icpPanel.dt Thr",0.01,0.01,0.001);

  pangolin::Var<bool> doSO3prealign("icpPanel.SO3 prealign",true,true);
  pangolin::Var<bool> useGpuPrealign("icpPanel.GPU prealign",true,true);
  pangolin::Var<float> SO3HThr("icpPanel.SO3 H Thr",-24.,-40.,-20.);
  pangolin::Var<float> SO3negLogEvThr("icpPanel.SO3 neg log ev Thr",-6.,-10.,0.);
  pangolin::Var<float> SO3condEntropyThr("icpPanel.SO3 rel log dH ", 1.e-3,1.e-6,1e-2);
  pangolin::Var<int> SO3maxIt("icpPanel.SO3 max iter",2, 1, 20);
  pangolin::Var<int> SO3maxLvl("icpPanel.SO3 max Lvl",PYR-1,0,PYR-1);
  pangolin::Var<int> SO3minLvl("icpPanel.SO3 min Lvl",1,0,PYR-1);

  pangolin::Var<float> renderPointSize("visPanel.pt size",1.5,0.1,10.);
  pangolin::Var<float> renderLineWidth("visPanel.line w",1.5,0.1,10.);
  pangolin::Var<float> scale("visPanel.scale",0.05,0.1,1);
  pangolin::Var<int>   step("visPanel.step",10,1,100);
  pangolin::Var<float> bgGrey("visPanel.bg Grey",0.2,0.0,1);
  pangolin::Var<float> showNumStdPose("visPanel.std pose",3.,0.0,6.);
  pangolin::Var<bool> showRgbView("visPanel.showRGBview",true,true);
  pangolin::Var<bool> showSurfaceNormalView("visPanel.showSurfaceNormalView",true,true);
  pangolin::Var<bool> showGradDir("visPanel.showGradDir",true,true);
  pangolin::Var<bool> showFullPc("visPanel.show full",true,true);
  pangolin::Var<bool> showNormals("visPanel.show ns",false,true);
  pangolin::Var<bool> showGrads("visPanel.show grads",false,true);
  pangolin::Var<bool> showSamplePcEst("visPanel.show SamplePcEst",false,true);
  pangolin::Var<bool> showSamplePc("visPanel.show SamplePc",false,true);
  pangolin::Var<bool> showPcMu("visPanel.show PcMu",false,true);
  pangolin::Var<float> showLow("visPanel.show low",0.,0.,0.);
  pangolin::Var<float> showHigh("visPanel.show high",0.,0.,0.);
  pangolin::Var<float> showLowPerc("visPanel.show lowPerc",0.1,0.,.5);
  pangolin::Var<float> showHighPerc("visPanel.show highPerc",0.9,0.000001,.5);
  pangolin::Var<float> showLowH("visPanel.show low H",-25,-30.,-15.0);
  pangolin::Var<float> showHighH("visPanel.show high H",-15,-15.,-5.0);
  pangolin::Var<bool> showHp("visPanel.show Hp",false,true);
  pangolin::Var<bool> showHn("visPanel.show Hn",false,true);
  pangolin::Var<bool> showAge("visPanel.show age",false,true);
  pangolin::Var<bool> showObs("visPanel.show # obs",false,true);
  pangolin::Var<bool> showCurv("visPanel.show curvature",false,true);
  pangolin::Var<bool> showGrey("visPanel.show grey",false,true);
  pangolin::Var<bool> showP2PlVar("visPanel.show p2pl var",false,true);
  pangolin::Var<bool> showIvar("visPanel.show I var",false,true);
  pangolin::Var<bool> showImean("visPanel.show I mean",false,true);
  pangolin::Var<bool> showRadius("visPanel.show Radius",false,true);
  pangolin::Var<bool> showNumSum("visPanel.show numSum",false,true);
  pangolin::Var<bool> showLabelCounts("visPanel.show LabelCount",false,true);
  pangolin::Var<bool> showNSampleCount("visPanel.show nSampleCount",false,true);
  pangolin::Var<bool> showNSamplePReject("visPanel.show nSample P Rej",false,true);
  pangolin::Var<bool> showLabels("visPanel.show Sample labels",false,true);
  pangolin::Var<bool> showLabelsMl("visPanel.show ML labels",true,true);
  pangolin::Var<bool> showSamples("visPanel.show Samples",false,true);
  pangolin::Var<bool> showSurfels("visPanel.show surfels",true,true);
  pangolin::Var<bool> showNN("visPanel.show NN",false,true);
  pangolin::Var<bool> showPcCurrent("visPanel.show current",false,true);
  pangolin::Var<int> showPcLvl("visPanel.cur Lvl",0,0,PYR-1);
  pangolin::Var<bool> showLoopClose("visPanel.show loopClose",false,true);

  pangolin::Var<float> ransacMaxIt("ui.max it",3000,1,1000);
  pangolin::Var<float> ransacThr("ui.thr",0.09,0.01,1.0);
  pangolin::Var<float> ransacInlierThr("ui.inlier thr",6,1,20);

  tdp::SE3f T_wc_0;
  tdp::SE3f T_wc = T_wc_0;
  tdp::SE3f T_wcRansac;
  std::vector<tdp::SE3f> T_wcs;
  Eigen::Matrix<float,6,6> Sigma_wc = Eigen::Matrix<float,6,6>::Identity()*1e-12;

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


  tdp::ManagedHostCircularBuffer<tdp::VectorZfda> zSampleCounts(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::VectorZuda> zSampleIds(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<uint16_t> zMl(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> zMlCount(MAP_SIZE);
  zSampleCounts.Fill(tdp::VectorZfda::Zero());
  zSampleIds.Fill(tdp::VectorZuda::Ones()*999);
  zMl.Fill(0);
  zMlCount.Fill(0);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nSampleSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Matrix3fda> pSampleOuter_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Matrix3fda> pSampleCov_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pSampleSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pSampleEst_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> pSampleCount(MAP_SIZE);
  pSampleOuter_w.Fill(tdp::Matrix3fda::Zero());
  pSampleCov_w.Fill(tdp::Matrix3fda::Zero());
  pSampleSum_w.Fill(tdp::Vector3fda::Zero());
  pSampleEst_w.Fill(tdp::Vector3fda::Zero());
  nSampleSum_w.Fill(tdp::Vector3fda::Zero());
  pSampleCount.Fill(0);
  size_t pTotalSampleCount = 0;

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
  vboEst_w.Upload(pSampleEst_w.ptr_, pSampleEst_w.SizeBytes(), 0);

  tdp::ManagedHostCircularBuffer<uint8_t> nnFixed(MAP_SIZE);
  nnFixed.Fill(0);
  tdp::ManagedHostCircularBuffer<tdp::VectorkNNida> nn(MAP_SIZE);
  nn.Fill(tdp::VectorkNNida::Ones()*-1);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> numSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> normSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> tauOSum_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Matrix3fda> pcObsInfo_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pcObsXi_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pcObsMu_w(MAP_SIZE);

  tdp::ManagedHostCircularBuffer<float> p2plSum(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> p2plSqSum(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> p2plCount(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> p2plVar(MAP_SIZE);
  p2plSum.Fill(0.);
  p2plSqSum.Fill(0.);
  p2plCount.Fill(0.);
  p2plVar.Fill(sigmaPl*sigmaPl);

  tdp::ManagedHostCircularBuffer<float> ImSum(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> ImSqSum(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> ImCount(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> ImVar(MAP_SIZE);

  pcObsInfo_w.Fill(tdp::Matrix3fda::Zero());
  pcObsXi_w.Fill(tdp::Vector3fda::Zero());
  pcObsMu_w.Fill(tdp::Vector3fda::Zero());

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
      if (sizeToRead > sizeToReadPrev) {
        for (int32_t i=sizeToReadPrev; i<sizeToRead; ++i)
          newIds.push_back(i);
      }
      if (newIds.size() == 0) {
        std::uniform_int_distribution<int32_t> unif(0, sizeToRead-1);
        newIds.push_back(unif(rnd));
//        std::cout << "sampled next id :" << newIds.back() << "/" << sizeToRead << std::endl;
      }
      iReadNext = newIds.front();
      newIds.pop_front();
      idNNStack = newIds.size();

      if (allowNNRevisit || nnFixed[iReadNext] < kNN) {
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        if (!pl.valid_) continue;
        values.fill(std::numeric_limits<float>::max());
        tdp::VectorkNNida& ids = nn[iReadNext];
        tdp::VectorkNNida idsPrev = ids;
        ids = tdp::VectorkNNida::Ones()*(-1);

        if (pruneNoise) {
//          std::cout << "checking prune on " << iReadNext 
//            << " time " << pl.lastFrame_ << " now " << frame << std::endl;
          if (pl.lastFrame_+survivalTime < frame && pl.Hp_ > pruneHThr
              && pl.numObs_ < pruneNumObsThr) {
//          float dist = (pl.p_-pl_w[ids(0)].p_).squaredNorm();
////          std::cout << iReadNext << ": " 
////            << dist  <<  " vs " << values(0) 
////            << " frac " << dist/values(0) 
////            << std::endl;
//          if (dist/values(0) > relPruneDistThr
//              || (pSampleEst_w[iReadNext] - pl.p_).squaredNorm()/values(0) > relPruneDistThr ) {
            pc_w[iReadNext] = tdp::Vector3fda(NAN,NAN,NAN);
            n_w[iReadNext]  = tdp::Vector3fda(NAN,NAN,NAN);
            pl.valid_ = false;
            std::cout << "pruning " << iReadNext 
//              << " rel val " << dist/values(0)
              << " H " << pl.Hp_ << " HThr " << HThr 
//              << " thr " << relPruneDistThr 
              << std::endl;
//            std::cout << values.transpose() << std::endl;
            numPruned = numPruned +1;
//          }
            for (int32_t i=0; i<kNN; ++i) 
              mapNN[iReadNext*kNN+i] = std::pair<int32_t,int32_t>(iReadNext, -1);
            continue;
          }
        }

        TICK("full NN pass");
        for (int32_t i=0; i<sizeToRead; ++i) {
          if (i != iReadNext && pl_w[i].valid_) {
            float dist = (pl.p0_-pl_w[i].p0_).squaredNorm();
            tdp::AddToSortedIndexList<kNN>(ids, values, i, dist);
//            std::cout << i << ", " << dist << "| " <<  ids.transpose() << " : " << values.transpose() << std::endl;
          }
        }
        TOCK("full NN pass");

        // for map constraints
        // TODO: should be updated as pairs are reobserved
        nnFixed[iReadNext] = kNN;
        for (int32_t i=0; i<kNN; ++i) {
//            if (ids(i) != idsPrev(i)) {
//              numSamplesZ[iReadNext][i] = 0;
//              sumSameZ[iReadNext][i] = 0;
//            }
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
  tdp::ManagedHostCircularBuffer<uint32_t> nSampleCount(MAP_SIZE); // count how often the same cluster ID
  tdp::ManagedHostCircularBuffer<float> nSamplePReject(MAP_SIZE); // count how often the same cluster ID
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nS(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pS(MAP_SIZE);
  size_t nTotalSampleCount = 0;
  size_t nMaxSampleCount = 0;
  nSampleCount.Fill(0);
  nSamplePReject.Fill(NAN);
  nS.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  pS.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  zS.Fill(9999); //std::numeric_limits<uint32_t>::max());
  tdp::ManagedHostCircularBuffer<tdp::Vector4fda> vmfSS(10000);
  vmfSS.Fill(tdp::Vector4fda::Zero());

  std::thread samplingNormals([&]() {
    int32_t i = 0;
    int32_t sizeToReadPrev = 0;
    int32_t sizeToRead = 0;
    std::deque<int32_t> newIds;
//    std::random_device rd_;
    std::mt19937 rnd(19201420);
    std::uniform_real_distribution<float> coin(0, 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    while(runSampling.Get()) {
      if (i%100 == 0 || sizeToRead == 0) {
        {
          std::lock_guard<std::mutex> lock(nnLock); 
          sizeToRead = nn.iInsert_;
        }
      }
      if (sizeToRead ==0) continue;
      i = (i+1)% sizeToRead;
      if (sizeToRead > sizeToReadPrev) {
        i = sizeToReadPrev;
        sizeToReadPrev = sizeToRead;
//        nTotalSampleCount += sizeToRead;
      }
//      float pDontSample = ((float)nSampleCount[i]+alphaSchedule)/((float)nTotalSampleCount+alphaSchedule);
      float pDontSample = ((float)nSampleCount[i])/((float)nMaxSampleCount+alphaSchedule);
      nSamplePReject[i] = pDontSample;
      if (sampleScheduling && coin(rnd) < pDontSample)  {
//        std::cout << " normals skipping " << i << std::endl;
        continue;
      }
//      std::cout << " normals sampling " << i << std::endl;
      // sample normals using dpvmf and observations from planes
//      vmfSS.Fill(tdp::Vector4fda::Zero());
      TICK("sample normals");
      uint16_t& zi = zS[i];
      tdp::Vector3fda& ni = nS[i];
      if (!pl_w[i].valid_) 
        continue;
      tdp::Vector3fda mu;
      if (useSurfNormalObs) {
        mu = normSum_w[i]*nSum_w[i]*tauO;
      } else {
        mu = tdp::Vector3fda::Zero();
      }
      if (estimateTauO && useSurfNormalObs)
        mu = normSum_w[i]*nSum_w[i]*numSum_w[i]*tauOSum_w[i];
      if (zi < vmfs.size()) {
        mu += vmfs[zi].mu_*vmfs[zi].tau_;
      }
      if (normalsP2PlContrib) {
        tdp::VectorkNNida& ids = nn.GetCircular(i);
        if ((ids.array() >= 0).all()) {
          Eigen::Matrix3f Info = Eigen::Matrix3f::Zero();
          Eigen::Matrix3f InfoPl;
          int32_t num = 0;
          for (int k=0; k<kNN; ++k) {
            if (ids[k] > -1 
                && pl_w[ids[k]].valid_ 
                && tdp::IsValidData(pS[ids[k]])
                && tdp::IsValidData(pS[i])) {
              InfoPl = (pS[ids[k]]-pS[i])*(pS[ids[k]]-pS[i]).transpose();
              //                  if (i%10) {
              //                    std::cout << k << " " << (pS[ids[k]]-pS[i]).transpose() << std::endl;
              //                  }
              if (useSigmaPl) InfoPl *= 1./(sigmaPl*sigmaPl);
              Info += 0.5*InfoPl;
              num++;
            }  else {
              break;
            } 
          }
          if (num == kNN) {
            eig.computeDirect(Info);
            tdp::Vector3fda e = eig.eigenvalues();
            float tauEst = 4.*(e(1)*e(2))/(e(1)+e(2));
            tdp::Vector3fda muEst = eig.eigenvectors().col(0);

            if (!useSurfNormalObs) {
              muEst *= muEst.dot(nSum_w[i]) > 0? 1 : -1;
            } else {
              muEst *= muEst.dot(mu)/mu.norm() > 0? 1 : -1;
            }
            mu += muEst*tauEst;
          }
        }
      }
      ni = vMF<float,3>(mu).sample(rnd);
      nSampleSum_w[i] += ni;
      nSampleCount[i] ++;
      nTotalSampleCount ++;
      nMaxSampleCount = std::max(nMaxSampleCount, (size_t) nSampleCount[i]);
      if (nSampleCount[i] > sampleCountThr) {
        pl_w[i].n_ = nSampleSum_w[i].normalized();
        n_w[i] = pl_w[i].n_;
      }
      // TODO: could play with sync across threads here to never have
      // go over all SS;
//      if (zi == 9999) {
//        vmfSS[zS[i]].topRows<3>() += ni;
//        vmfSS[zS[i]](3) ++;
//      }
      TOCK("sample normals");
    }
  });

  std::thread sampling([&]() {
    int32_t iInsert = 0;
//    std::random_device rd_;
    std::mt19937 rnd(0);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    while(runSampling.Get()) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        iInsert = nn.iInsert_;
      }
      if (iInsert == 0) continue;
      TICK("sample labels");
      pS.iInsert_ = nn.iInsert_;
      nS.iInsert_ = nn.iInsert_;
      size_t Ksample = vmfs.size();
      vmfSS.Fill(tdp::Vector4fda::Zero());
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        if (!pl_w[i].valid_ || !tdp::IsValidData(nS[i])) continue;
        vmfSS[zS[i]].topRows<3>() += nS[i];
        vmfSS[zS[i]](3) ++;
      }
//      TOCK("sample normals");
      // sample dpvmf labels
      for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
        tdp::Vector3fda& ni = nS[i];
        if (!pl_w[i].valid_ || !tdp::IsValidData(ni)) continue;
        Eigen::VectorXf logPdfs(Ksample+1);
        Eigen::VectorXf pdfs(Ksample+1);

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
        }
      }
      TOCK("sample labels");
      TICK("sample params");
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
        uint16_t zMli = 0;
        float countMli = 0;
        for (int32_t i = 0; i!=iInsert; i=(i+1)%nn.w_) {
          if (!pl_w[i].valid_ || zS[i] >= K) continue;
          tdp::InsertLabelML(zSampleIds[i], zSampleCounts[i], zS[i],
              zMli, countMli);
//          if (i%100) 
//            std::cout << zSampleIds[i].transpose() << std::endl << zSampleCounts[i].transpose() << std::endl 
//              << "ML: " << zMli << " " << countMli << " zi=" << zS[i] << std::endl;
          for (uint32_t k=0; k<zKTrac; ++k) {
            if (zSampleIds[i](k) < K)
              zSampleIds[i](k) = labelMap[zSampleIds[i](k)];
          }
          zS[i] = labelMap[zS[i]];
          pl_w[i].z_ = labelMap[zMli];
          zMl[i] = pl_w[i].z_;
          zMlCount[i] = countMli;
        }
        K = Ksample;
      }
      TOCK("sample params");
      TOCK("sample full");
    };
  });

  std::thread samplingPoints([&]() {
    int32_t i = 0;
    int32_t sizeToReadPrev = 0;
    int32_t sizeToRead = 0;
    std::deque<int32_t> newIds;
    std::mt19937 rnd(3839129);
    std::uniform_real_distribution<float> coin(0, 1);
    // sample points
    while(runSampling.Get()) {
      if (i%100 == 0 || sizeToRead == 0) {
        {
          std::lock_guard<std::mutex> lock(nnLock); 
          sizeToRead = nn.iInsert_;
        }
      }
      if (sizeToRead ==0) continue;
      i = (i+1)% sizeToRead;
      if (sizeToRead > sizeToReadPrev) {
        i = sizeToReadPrev;
        sizeToReadPrev = sizeToRead;
//        pTotalSampleCount+=sizeToRead;
      }
      float pDontSample = ((float)pSampleCount[i]+alphaSchedule)/((float)pTotalSampleCount+alphaSchedule);
      if (sampleScheduling && coin(rnd) < pDontSample)  {
        continue;
      }

      tdp::Plane& pl = pl_w[i];
      if (!pl.valid_) continue;
      TICK("sample points");
      tdp::VectorkNNida& ids = nn.GetCircular(i);
      bool haveFullNeighborhood = (ids.array() >= 0).all();
      if (haveFullNeighborhood && numSum_w[i] > 0) {
        Eigen::Matrix3f Info = pcObsInfo_w[i]; 
        Eigen::Vector3f xi =   pcObsXi_w[i]; 
        Eigen::Matrix3f InfoPl;
        Eigen::Matrix3f InfoPlSum = Eigen::Matrix3f::Zero();
        Eigen::Vector3f xiPl = Eigen::Vector3f::Zero();
        uint32_t numNN = 0;
        for (int k=0; k<kNN; ++k) {
          if (ids[k] > -1 
              && pl_w[ids[k]].valid_ 
              && tdp::IsValidData(pS[ids[k]])
              && tdp::IsValidData(pS[i])) {
            if (useOtherNi) {
              InfoPl = pl_w[ids[k]].n_*pl_w[ids[k]].n_.transpose();
            } else {
              InfoPl = pl_w[i].n_*pl_w[i].n_.transpose();
            }
            if (useSigmaPl) InfoPl *= 1./(sigmaPl*sigmaPl);
            InfoPlSum += 0.5*InfoPl;
            xiPl += InfoPl*pS[ids[k]];
            numNN++;
          } else {
            break;
          }
        } 
        if (numNN == kNN) {
          Info += InfoPlSum;
          xi += xiPl;
        }
        if ((Info.eigenvalues().real().array() < 0.).any() ) {
          std::cout <<  "low Information! "  << numNN << " neighs "
            << numSum_w[i] 
            << " obs, evs " << Info.eigenvalues().transpose()
            //              << " mu " << mu.transpose()
            << " xi " << xi.transpose() << std::endl;
          pl.valid_ = false;
          pc_w[i] = tdp::Vector3fda(NAN,NAN,NAN);
          n_w[i] = tdp::Vector3fda(NAN,NAN,NAN);
        }
        Eigen::Matrix3f Sigma = Info.inverse();
        Eigen::Vector3f mu = Info.ldlt().solve(xi);
        //        std::cout << xi.transpose() << " " << mu.transpose() << std::endl;
        tdp::Vector3fda& pi = pS[i];
        pi = Normal<float,3>(mu, Sigma).sample(rnd);
        pSampleSum_w[i] += pi;
        pSampleOuter_w[i] += pi*pi.transpose();
        pSampleCount[i] ++;
        pTotalSampleCount ++;

        for (int k=0; k<kNN; ++k) {
          if (ids[k] > -1 
              && pl_w[ids[k]].valid_ 
              && tdp::IsValidData(pS[ids[k]])
              && tdp::IsValidData(pS[i])) {
            float p2pl = nS[i].dot(pS[ids[k]] - pS[i]);
            p2plSum[i] += p2pl;
            p2plSqSum[i] += p2pl*p2pl;
            p2plCount[i] ++;
          }
        }
        if (p2plCount[i] > 0) {
          p2plVar[i] = (p2plSqSum[i] - p2plSum[i]*p2plSum[i]/p2plCount[i])/p2plCount[i];
        }

        if (false && i%10) {
          std::cout << pSampleCount[i] << ": " << pi.transpose() << "; " << pSampleSum_w[i].transpose() << std::endl;
          std::cout << Sigma << std::endl;
          std::cout << Info << std::endl;
          std::cout << xi.transpose() << std::endl;
          std::cout << mu.transpose() << std::endl;
        }

        pSampleCov_w[i] = (pSampleOuter_w[i] - pSampleSum_w[i]*pSampleSum_w[i].transpose()/(float)pSampleCount[i])/(float)pSampleCount[i];
        float Hp = 0.5*pSampleCov_w[i].eigenvalues().real().array().log().sum();
        pSampleEst_w[i] = pSampleSum_w[i]/(float)pSampleCount[i];
        //          if (i%10) {
        //            std::cout << "Hp " << Hp << " dH " << Hp - pl.Hp_ << std::endl;
        //          }
        if (samplePoints) {
          if ( fabs(Hp - pl.Hp_) < condHThr 
              && pSampleCount[i] > sampleCountThr
              && pl.numObs_ > sampleCountThr) {
            //            if (pSampleCount[i] > 30)
            pl.p_ = pSampleEst_w[i];
          }
          pc_w[i] = pl.p_;
          if (pSampleCount[i] > 3) {
            pl.Hp_ = Hp;
          }
        }
        //          pl.Hp_ = -Info.eigenvalues().real().array().log().sum();
      }
      idMapUpdate = i;
      TOCK("sample points");
//      pS.iInsert_ = sizeToRead;
//      nS.iInsert_ = sizeToRead;
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

  if (varsMapFile.size() > 0)
    pangolin::LoadJsonFile(varsMapFile, "mapPanel");
  if (varsIcpFile.size() > 0)
    pangolin::LoadJsonFile(varsIcpFile, "icpPanel");

  pangolin::SaveJsonFile("./varsUi.json", "ui");
  pangolin::SaveJsonFile("./varsMap.json", "mapPanel");
  pangolin::SaveJsonFile("./varsVis.json", "visPanel");
  pangolin::SaveJsonFile("./varsIcp.json", "icpPanel");

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (showVisPanel.GuiChanged()) {
      pangolin::Display("visPanel").Show(showVisPanel);
    }
    if (showMapPanel.GuiChanged()) {
      pangolin::Display("mapPanel").Show(showMapPanel);
    }
    if (showIcpPanel.GuiChanged()) {
      pangolin::Display("icpPanel").Show(showIcpPanel);
    }
    if (showSurfaceNormalView.GuiChanged()) viewNormals.Show(showSurfaceNormalView);
    if (showRgbView.GuiChanged()) viewCurrent.Show(showRgbView);

    if (frame == 30) {
      sampleScheduling = true;
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
      if (sigmaOclusion) {
        projAssoc.GetAssocOcclusion(pl_w, pSampleCov_w, pyrPc, pyrRay,
            T_wc.Inverse(), numSigmaOclusion, dMin, dMax, pyrZ,
            pyrMask, idsCur);
      } else {
        projAssoc.GetAssocOcclusion(pl_w, pyrPc, T_wc.Inverse(),
            occlusionDepthThr, dMin, dMax, pyrZ, pyrMask, idsCur);
      }
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
             mask, W, frame, T_wc, Sigma_wc, cam, rho, rays, outerRaysInt,
             dpc, pl_w, pc_w, pcObsInfo_w,
             pSampleCov_w, rgb_w, n_w, grad_w, ImSum, ImSqSum, ImCount,
             ImVar, rs, ts, normalExtractMethod, useTrackingUncertainty);

        std::cout << " extracted " << pl_w.iInsert_-iReadCurW << " new planes " << std::endl;
        TOCK("normals");
        TICK("add to model");
        for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
          gradDir_w[i] = pl_w[i].grad_.normalized();
         
          pcObsInfo_w[i] /= obsStdInflation*obsStdInflation;
          pcObsXi_w[i] = pcObsInfo_w[i]*pl_w[i].p_;
          pcObsMu_w[i] = pcObsInfo_w[i].ldlt().solve(pcObsXi_w[i]);

          nSum_w[i] = pl_w[i].n_;
          numSum_w[i] = 1;
          normSum_w[i] = 1;
          tauOSum_w[i] = pl_w[i].curvature_;
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
                if (k < invInd[lvl]->size() && invInd[lvl]->at(k).size() < 3000)
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
                if (k < invInd[lvl]->size() && invInd[lvl]->at(k).size() < 3000)
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
    if (normalExtractMethod) {
      tdp::ConvertDepthToInverseDepthGpu(cuPyrD, cuPyrRho);
      pyrRho.CopyFrom(cuPyrRho);
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
        Eigen::Matrix<float,6,6> Ap2pl;
        Eigen::Matrix<float,6,1> bp2pl;
        Eigen::Matrix<float,6,6> Aphoto;
        Eigen::Matrix<float,6,1> bphoto;
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
          tdp::Image<float> rhoLvl = pyrRho.GetImage(pyr);
          tdp::Image<float> greyFlLvl = pyrGreyFl.GetImage(pyr);
          tdp::Image<float> greyFlPrevLvl = pyrGreyFlPrev.GetImage(pyr);
          tdp::Image<tdp::Vector2fda> gradGreyLvl = pyrGradGrey.GetImage(pyr);
          tdp::Image<tdp::Vector3fda> pcLvl = pyrPc.GetImage(pyr);
          tdp::Image<tdp::Vector3fda> nLvl = pyrN.GetImage(pyr);
          tdp::Image<tdp::Vector3fda> rayLvl = pyrRay.GetImage(pyr);
          tdp::Image<tdp::Vector6dda> outerRaysIntLvl = pyrOuterRaysInt.GetImage(pyr);
          if (gui.verbose) std::cout << "pyramid lvl " << pyr << " scale " << scale << std::endl;
          for (size_t it = 0; it < maxItLvl[pyr]; ++it) {
            TICK("icp it");
            for (auto& ass : assoc) mask[ass.second] = 0;
            assoc.clear();
            indK = std::vector<size_t>(invInd[pyr]->size(),0);
            numProjected = 0;
            Ap2pl = Eigen::Matrix<float,6,6>::Zero();
            bp2pl = Eigen::Matrix<float,6,1>::Zero();
            Aphoto = Eigen::Matrix<float,6,6>::Zero();
            bphoto = Eigen::Matrix<float,6,1>::Zero();
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
              TICK("icp one pt");
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
                int32_t u = floor(x(0)+0.5f);
                int32_t v = floor(x(1)+0.5f);
                float d_c = dLvl.GetBilinear(x(0),x(1));
//                std::cout << d_c << std::endl;
                float threeSigma_d = numSigmaOclusion*sqrtf(rayLvl(u,v).dot(pSampleCov_w[i]*rayLvl(u,v)));
                if (d_c != d_c 
                    || (!sigmaOclusion && fabs(d_c-pc_w_in_c(2)) > occlusionDepthThr)
                    || (sigmaOclusion  && fabs(d_c-pc_w_in_c(2)) > threeSigma_d)
                    )
                  continue;
                numProjected = numProjected + 1;
                if (!EnsureNormal(pcLvl, dpc, rhoLvl, rayLvl,
                      outerRaysIntLvl, W, nLvl, curv, rad, u, v,
                      normalExtractMethod))
                  //              if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, camLvl, pc,
                  //                    W, dpc, n, curv, u,v ))
                  continue;
                if (useTexture) {
                  float sqrtInfoP2Pl = estSigmaPl ? lambdaP2Pl/sqrtf(p2plVar[i]) : 1./sigmaPl;
                  float sqrtInfoIm = estSigmaIm ? lambdaTex/sqrtf(ImVar[i]) : lambdaTex/sigmaPl;
                  if (!AccumulateP2PlIntensity(pl, T_wc, T_cw, camLvl, pcLvl(u,v),
                        nLvl(u,v), greyFlLvl(u,v), gradGreyLvl(u,v), p2plThr, dotThr,
                        sqrtInfoP2Pl, sqrtInfoIm, Ap2pl, bp2pl, Aphoto, bphoto, Ai, err))
                    continue;
                  A = Aphoto + Ap2pl;
                  b = bphoto + bp2pl;
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
              TOCK("icp one pt");
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
            TOCK("icp it");
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

//        if (useTrackingUncertainty) {
//          Sigma_wc = Ap2pl.inverse();
//        } else {
          Sigma_wc = A.inverse();
//        }
        std::cout << "Sigma_wc" << std::endl << Sigma_wc << std::endl;

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
          SigmaO *= obsStdInflation*obsStdInflation;
          if (useTrackingUncertainty) {
            tdp::InflateObsCovByTransformationCov(T_wc, pl.p_, Sigma_wc, SigmaO);
          }
          SigmaO = T_wc.rotation().matrix()*SigmaO*T_wc.rotation().matrix().transpose();

          ts[i] = frame;
          pl.lastFrame_ = frame;
          pl.numObs_ ++;

          float w = numSum_w[i];
          // filtering grad grey
          pl.grad_ = (pl.grad_*w + pl.Compute3DGradient(T_wc, cam, u, v, gradGrey(u,v)))/(w+1);
          pl.grey_ = (pl.grey_*w + greyFl(u,v)) / (w+1);
          pl.gradNorm_ = (pl.gradNorm_*w + gradGrey(u,v).norm()) / (w+1);
          pl.rgb_ = ((pl.rgb_.cast<float>()*w + rgb(u,v).cast<float>()) / (w+1)).cast<uint8_t>();
          pl.r_ = std::min(pl.r_, rad(u,v));
          rs[i] = pl.r_;

          tdp::Matrix3fda infoObs = SigmaO.inverse();
          tdp::Vector3fda xiObs = infoObs*pc_c_in_w;
//          pcObsInfo_w[i] = (pcObsInfo_w[i]*w + infoObs)/(w+1.);
//          pcObsXi_w[i] = (pcObsXi_w[i]*w + xiObs)/(w+1.);
            pcObsInfo_w[i] = pcObsInfo_w[i] + infoObs;
            pcObsXi_w[i] = pcObsXi_w[i] + xiObs;
            pcObsMu_w[i] = pcObsInfo_w[i].ldlt().solve(pcObsXi_w[i]);


          nSum_w[i] = (nSum_w[i]*w + n_c_in_w)/(w+1.);
          normSum_w[i] = nSum_w[i].norm();
          nSum_w[i] /= nSum_w[i].norm();
          tauOSum_w[i] = (tauOSum_w[i]*w + curv(u,v))/(w+1.);
          numSum_w[i] = std::min(50.f, w+1.f);

          grad_w[i] = pl.grad_;
          gradDir_w[i] = grad_w[i].normalized();

            if (ProjectiveAssocOcl(pS[i], T_wc, cam, d,
                  occlusionDepthThr, u, v)) {
              ImSum[i] += greyFl(u,v);
              ImSqSum[i] += greyFl(u,v)*greyFl(u,v);
              ImCount[i] ++;
              ImVar[i] = (ImSqSum[i] - ImSum[i]*ImSum[i]/ImCount[i])/ImCount[i];
            }

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
            int32_t u, v;
            if (!ProjectiveAssocOcl(pl, T_wc, cam, d, occlusionDepthThr, u, v))
              continue;
            if (!EnsureNormal(pc, dpc, rho, rays, outerRaysInt, W, n,
                  curv, rad, u, v, normalExtractMethod))
              continue;
//            std::cout << " adding " << j << " of " << numAdditionalObs 
//              << " id " << i << " uv " << u << "," << v << std::endl;
            numProjected = numProjected + 1;
            tdp::Vector3fda n_c_in_w = T_wc.rotation()*n(u,v);
            tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);

            tdp::Matrix3fda SigmaO;
            NoiseModelNguyen(n(u,v), pc(u,v), cam, SigmaO);
            SigmaO *= obsStdInflation*obsStdInflation;
            if (useTrackingUncertainty) {
              tdp::InflateObsCovByTransformationCov(T_wc, pl.p_, Sigma_wc, SigmaO);
            }
            SigmaO = T_wc.rotation().matrix()*SigmaO*T_wc.rotation().matrix().transpose();

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

            pl.r_ = std::min(pl.r_, rad(u,v));
            rs[i] = pl.r_;

            tdp::Matrix3fda infoObs = SigmaO.inverse();
            tdp::Vector3fda xiObs = infoObs*pc_c_in_w;
//            pcObsInfo_w[i] = (pcObsInfo_w[i]*w + infoObs)/(w+1.);
//            pcObsXi_w[i] = (pcObsXi_w[i]*w + xiObs)/(w+1.);
            pcObsInfo_w[i] = pcObsInfo_w[i] + infoObs;
            pcObsXi_w[i] = pcObsXi_w[i] + xiObs;
            pcObsMu_w[i] = pcObsInfo_w[i].ldlt().solve(pcObsXi_w[i]);

            nSum_w[i] = (nSum_w[i]*w + n_c_in_w)/(w+1.);
            normSum_w[i] = nSum_w[i].norm();
            nSum_w[i] /= nSum_w[i].norm();
            tauOSum_w[i] = (tauOSum_w[i]*w + curv(u,v))/(w+1.);
            numSum_w[i] = std::min(50.f, w+1.f);

            grad_w[i] = pl.grad_;
            gradDir_w[i] = grad_w[i].normalized();

            if (ProjectiveAssocOcl(pS[i], T_wc, cam, d,
                  occlusionDepthThr, u, v)) {
              ImSum[i] += greyFl(u,v);
              ImSqSum[i] += greyFl(u,v)*greyFl(u,v);
              ImCount[i] ++;
              ImVar[i] = (ImSqSum[i] - ImSum[i]*ImSum[i]/ImCount[i])/ImCount[i];
            }

            if (!updateMap) {
              pl.AddObs(pc_c_in_w, n_c_in_w);
              n_w[i] =  pl.n_;
              pc_w[i] = pl.p_;
            }
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

    if (!gui.paused())
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
    glPointSize(renderPointSize);
    glLineWidth(renderLineWidth);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);
      glClearColor(bgGrey, bgGrey, bgGrey, 1.0f);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glColor4f(0.,1.,0.,1.0);
//      pangolin::glDrawAxis(T_wc.matrix(), 0.05f);
      pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wc.matrix(), 0.1f);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,3,3>> eig(Sigma_wc.bottomRightCorner<3,3>());
      pangolin::glSetFrameOfReference(T_wc.matrix());
      glColor4f(1.,0.,0.,1.0);
      Eigen::Vector3f std0 = eig.eigenvectors().col(0)*sqrtf(eig.eigenvalues()(0))*showNumStdPose;
      Eigen::Vector3f std1 = eig.eigenvectors().col(1)*sqrtf(eig.eigenvalues()(1))*showNumStdPose;
      Eigen::Vector3f std2 = eig.eigenvectors().col(2)*sqrtf(eig.eigenvalues()(2))*showNumStdPose;
      Eigen::Vector3f std0neg = -std0;
      Eigen::Vector3f std1neg = -std1;
      Eigen::Vector3f std2neg = -std2;
      tdp::glDrawLine(Eigen::Vector3f::Zero(), std0);
      tdp::glDrawLine(Eigen::Vector3f::Zero(),std0neg);
      tdp::glDrawLine(Eigen::Vector3f::Zero(), std1);
      tdp::glDrawLine(Eigen::Vector3f::Zero(),std1neg);
      tdp::glDrawLine(Eigen::Vector3f::Zero(), std2);
      tdp::glDrawLine(Eigen::Vector3f::Zero(),std2neg);
      pangolin::glUnsetFrameOfReference();


      if (showLoopClose) {
        glColor4f(1.,0.,0.,1.0);
        //      pangolin::glDrawAxis(T_wcRansac.matrix(), 0.05f);
        pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wcRansac.matrix(), 0.1f);
      }
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs, 100000, 0.03f);

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
        if (showAge || showObs || showCurv || showGrey || showNumSum ||
            showNSampleCount || showNSamplePReject || showHn || showHp
            || showP2PlVar || showIvar || showImean || showRadius ||
            showLabelCounts) {
          float min, max;
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
          } else if (showLabelCounts) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = zMlCount[i];
          } else if (showNSampleCount) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = nSampleCount[i];
          } else if (showNSamplePReject) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = nSamplePReject[i];
          } else if (showP2PlVar) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = sqrtf(p2plVar[i]);
          } else if (showIvar) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = sqrtf(ImVar[i]);
          } else if (showImean) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = ImSum[i] / ImCount[i];
          } else if (showRadius) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w[i].r_;
          } else if (showHp) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i)  {
              age[i] = pl_w[i].Hp_;
              max = std::max(max, pl_w[i].Hp_ < 0? pl_w[i].Hp_ : max);
            }
          } else if (showHn) {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) {
              age[i] = pl_w[i].Hn_;
            }
          } else {
            for (size_t i=0; i<pl_w.SizeToRead(); ++i) 
              age[i] = pl_w.GetCircular(i).curvature_;
          }
          valuebo.Upload(age.ptr_, pl_w.SizeToRead()*sizeof(float), 0);
          std::pair<float,float> minMaxAge = age.GetRoi(0,0,
              pl_w.SizeToRead(),1).MinMax();
          if (showHp){ 
            minMaxAge.first = showLowH;
            minMaxAge.second = showHighH;
          }
          std::cout << "drawn values are min " << minMaxAge.first 
            << " max " << minMaxAge.second << std::endl;
          showLow = minMaxAge.first;
          showHigh = minMaxAge.second;
          tdp::RenderVboValuebo(vbo_w, valuebo, 
              minMaxAge.first+showLowPerc*(minMaxAge.second-minMaxAge.first),
              minMaxAge.first+showHighPerc*(minMaxAge.second-minMaxAge.first), 
              P, MV);
          if (!showHp){ 
            showLow  = minMaxAge.first+showLowPerc*(minMaxAge.second-minMaxAge.first);
            showHigh = minMaxAge.first+showHighPerc*(minMaxAge.second-minMaxAge.first);
          }
        } else if (showLabels && frame > 1) {
          lbo.Upload(zS.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
          tdp::RenderLabeledVbo(vbo_w, lbo, s_cam);
        } else if (showLabelsMl && frame > 1) {
          lbo.Upload(zMl.ptr_, pl_w.SizeToRead()*sizeof(uint16_t), 0);
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
              glColor4f(0.3,0.3,0.3,0.3);
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
      if (showSamplePc) {
        vboEst_w.Upload(pS.ptr_, pl_w.SizeToRead()*sizeof(tdp::Vector3fda), 0);
        glColor3f(0.,1.,1.);
        pangolin::RenderVbo(vboEst_w);
        glColor4f(1,0,1,0.5);
        for (size_t i=0; i<pl_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w[i], pS[i]);
        }
      }
      if (showSamplePcEst) {
        vboEst_w.Upload(pSampleEst_w.ptr_, pl_w.SizeToRead()*sizeof(tdp::Vector3fda), 0);
        glColor3f(0.,1.,1.);
        pangolin::RenderVbo(vboEst_w);
        glColor4f(1,0,1,0.5);
        for (size_t i=0; i<pl_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w[i], pSampleEst_w[i]);
        }
      }
      if (showPcMu) {
        vboEst_w.Upload(pcObsMu_w.ptr_, pl_w.SizeToRead()*sizeof(tdp::Vector3fda), 0);
        glColor3f(0.,1.,1.);
        pangolin::RenderVbo(vboEst_w);
        glColor4f(1,0,1,0.5);
        for (size_t i=0; i<pl_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w[i], pcObsMu_w[i]);
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
          if (vmfSS[k](3) > 0 && vmfs[k].tau_ > 0) {
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
    if (pangolin::Pushed(snapShot)) {
      std::string name = tdp::MakeUniqueFilename("sparseFusion.png");
      name = std::string(name.begin(), name.end()-4);
      gui.container().SaveOnRender(name);
    }
    if (pangolin::Pushed(record)) {
      pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=30,bps=67108864,unique_filename]//screencap.avi");
    }

    if (gui.verbose) std::cout << "finished one iteration" << std::endl;
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();

    if (!gui.finished() && !gui.paused()) {
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

