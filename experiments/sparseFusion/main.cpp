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
#include <tdp/preproc/curvature.h>
#include <tdp/geometry/cosy.h>
#include <tdp/gl/shaders.h>
#include <tdp/utils/colorMap.h>
#include <tdp/camera/photometric.h>
#include <tdp/clustering/dpmeans_simple.hpp>
#include <tdp/features/brief.h>
#include <tdp/features/fast.h>
#include <tdp/preproc/blur.h>
#include <tdp/gl/render.h>
#include <tdp/preproc/convert.h>
#include <tdp/utils/timer.hpp>
#include <tdp/camera/projective_labels.h>

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

namespace tdp {

struct Plane {
  Vector3fda p_; 
  Vector3fda n_; 
  Vector3bda rgb_; 
  float grey_;
  Vector2fda gradGrey_; 
  Vector3fda dir_; 

  Brief feat_;

  uint32_t lastFrame_;
  uint32_t numObs_;

  float w_; // weight

  void AddObs(const Vector3fda& p, const Vector3fda& n) {
    float wNew = w_+1; 
    n_ = (n_*w_ + n).normalized();
    p_ += n.dot(p- p_)/wNew * n;
    w_ = std::min(100.f, wNew);
  }

  void AddObs(const Vector3fda& p, const Vector3fda& n, 
      const Vector3bda& rgb) {
    float wNew = w_+1; 
    p_ = (p_*w_ + p)/wNew;
    n_ = (n_*w_ + n).normalized();
    rgb_ = ((rgb_.cast<float>()*w_ + rgb.cast<float>())/wNew).cast<uint8_t>();
    w_ = std::min(100.f, wNew);
  }

  tdp::SE3f LocalCosy() {
    Eigen::Matrix3f R = tdp::OrthonormalizeFromYZ(
        dir_, n_);
    return tdp::SE3f(R, p_); 
  }


  bool Close(const Plane& other, float dotThr, float distThr, 
      float p2plThr) {
    if ((p_ - other.p_).norm() < distThr) {
      if (fabs(n_.dot(other.n_)) > dotThr) {
        if (fabs(n_.dot(p_ - other.p_)) < p2plThr) {
          return true;
        }
      }
    }
    return false;
  }

};

void UniformResampleEmptyPartsOfMask(
    const Image<Vector3fda>& pc, 
    const CameraT& cam,
    Image<uint8_t>& mask, 
    uint32_t W,
    float subsample,
    std::mt19937& gen,
    size_t I, 
    size_t J,
    size_t w,
    size_t h
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      float dSum = 0, numD = 0;
      for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
        for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
          if (mask(u,v)) count++;
          if (IsValidData(pc(u,v))) {
            dSum += pc(u,v)(2);
            numD ++;
          }
        }
      }
      const float avgD = dSum/numD;
//      const float area1 = I/cam.params_(0)*J/cam.params_(1);
//      const float areaEst = avgD*avgD*area1;
//      float prob = subsample*areaEst/area1;
      float prob = subsample*avgD*avgD;
      if (count == 0) {
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            if (coin(gen) < prob) {
              mask(u,v) = 1;
            }
          }
        }
      } else {
        for (size_t u=i*w/I; u<(i+1)*w/I; ++u) {
          for (size_t v=j*h/J; v<(j+1)*h/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

void UniformResampleEmptyPartsOfMask(
    Image<uint8_t>& mask, uint32_t W,
    float subsample,
    std::mt19937& gen,
    size_t I, 
    size_t J 
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) count++;
        }
      }
      if (count == 0) {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (coin(gen) < subsample) {
              mask(u,v) = 1;
            }
          }
        }
      } else {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

void UniformResampleMask(
    const Image<Vector3fda>& pc, 
    const CameraT& cam,
    Image<uint8_t>& mask, 
    uint32_t W,
    float subsample,
    std::mt19937& gen,
    size_t I, 
    size_t J 
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      float dSum = 0, numD = 0;
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) count++;
          if (IsValidData(pc(u,v))) {
            dSum += pc(u,v)(2);
            numD ++;
          }
        }
      }
      const float avgD = dSum/numD;
//      const float area1 = I/cam.params_(0)*J/cam.params_(1);
//      const float areaEst = avgD*avgD*area1;
//      float prob = subsample*areaEst/area1;
      float prob = subsample*avgD*avgD -(float)count/(float)(mask.w_/I*mask.h_/J);
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) {
            mask(u,v) = 0;
          } else if (coin(gen) < prob) {
            mask(u,v) = 1;
          } else {
            mask(u,v) = 0;
          }
        }
      }
    }
  }
}
 
void UniformResampleMask(
    Image<uint8_t>& mask, uint32_t W,
    float subsample,
    std::mt19937& gen,
    size_t I, 
    size_t J 
    ) {
  std::uniform_real_distribution<> coin(0, 1);
  for (size_t i=0; i<I; ++i) {
    for (size_t j=0; j<J; ++j) {
      size_t count = 0;
      for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
        for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
          if (mask(u,v)) count++;
        }
      }
      float perc = (float)subsample-(float)count/(float)(mask.w_/I*mask.h_/J);
      std::cout << i << "," << j << ": " << 100*perc 
        << ", " << count << std::endl;
      if (perc > 0.) {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (mask(u,v)) {
              mask(u,v) = 0;
            } else if (coin(gen) < perc) {
              mask(u,v) = 1;
            } else {
              mask(u,v) = 0;
            }
          }
        }
      } else {
        for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
          for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
            if (mask(u,v)) mask(u,v) = 0;
          }
        }
      }
    }
  }
}

void ExtractNormals(const Image<Vector3fda>& pc, 
    const Image<Vector3bda>& rgb,
    const Image<uint8_t>& grey,
    const Image<float>& greyFl,
    const Image<Vector2fda>& gradGrey,
    const Image<uint8_t>& mask, uint32_t W, size_t frame,
    const SE3f& T_wc, 
    ManagedHostCircularBuffer<Plane>& pl_w,
    ManagedHostCircularBuffer<Vector3fda>& pc_w,
    ManagedHostCircularBuffer<Vector3bda>& rgb_w,
    ManagedHostCircularBuffer<Vector3fda>& n_w) {
  Plane pl;
  tdp::Brief feat;
  Vector3fda n;
  for (size_t i=0; i<mask.Area(); ++i) {
    if (mask[i] && tdp::IsValidData(pc[i])
        && pc[i].norm() < 5. 
        && 0.3 < pc[i].norm() )  {
//      uint32_t Wscaled = floor(W*pc[i](2));
      uint32_t Wscaled = W;
//      if (tdp::NormalViaScatter(pc, i%mask.w_, i/mask.w_, Wscaled, n)) {
      if (tdp::NormalViaVoting(pc, i%mask.w_, i/mask.w_, Wscaled, 0.5, n)) {
        feat.pt_(0) = i%mask.w_;
        feat.pt_(1) = i/mask.w_;
        feat.orientation_ = ComputePatchOrientation(grey, feat.pt_);
        if (!tdp::ExtractBrief(grey, feat)) 
          continue;
        pl.p_ = T_wc*pc[i];
        pl.n_ = T_wc.rotation()*n;
        pl.rgb_ = rgb[i];
        pl.gradGrey_ = gradGrey[i];
        pl.grey_ = greyFl[i];
        pl.lastFrame_ = frame;
        pl.numObs_ = 1;
        pl.feat_ = feat;

        pl_w.Insert(pl);
        pc_w.Insert(pl.p_);
        n_w.Insert(pl.n_);
        rgb_w.Insert(pl.rgb_);
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
    const Image<Vector3fda>& pc,
    uint32_t W,
    Image<Vector3fda>& n,
    int32_t u,
    int32_t v
    ) {
  if (0 <= u && u < pc.w_ && 0 <= v && v < pc.h_) {
    if (tdp::IsValidData(pc(u,v))) {
//      uint32_t Wscaled = floor(W*pc(u,v)(2));
      uint32_t Wscaled = W;
      tdp::Vector3fda ni = n(u,v);
      if (!tdp::IsValidData(ni)) {
//        if(tdp::NormalViaScatter(pc, u, v, Wscaled, ni)) {
        if(tdp::NormalViaVoting(pc, u, v, Wscaled, 0.5, ni)) {
          n(u,v) = ni;
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
    const Image<Vector3fda>& pc,
    uint32_t W,
    Image<Vector3fda>& n,
    int32_t& u,
    int32_t& v
    ) {
  const tdp::Vector3fda& n_w =  pl.n_;
  const tdp::Vector3fda& pc_w = pl.p_;
  Eigen::Vector2f x = cam.Project(T_cw*pc_w);
  u = floor(x(0)+0.5f);
  v = floor(x(1)+0.5f);
  return EnsureNormal(pc, W, n, u, v);
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
  tdp::ImuInterface* imu = nullptr; 
  if (imu_input_uri.size() > 0) 
    imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();
  tdp::ImuInterpolator imuInterp(imu,nullptr);
  imuInterp.Start();

  tdp::GuiBase gui(1200,800,video);
  gui.container().SetLayout(pangolin::LayoutEqual);

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,false);
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
      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
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
  gui.container().AddDisplay(viewNormals);

//  pangolin::View& viewInternal = pangolin::CreateDisplay()
//    .SetHandler(new pangolin::Handler3D(s_cam));
//  gui.container().AddDisplay(viewInternal);

  tdp::QuickView viewInternal(w, h);
  gui.container().AddDisplay(viewInternal);

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewCurrent(wc, hc);
  containerTracking.AddDisplay(viewCurrent);
  tdp::QuickView viewMask(wc, hc);
  containerTracking.AddDisplay(viewMask);
  gui.container().AddDisplay(containerTracking);

  containerTracking.Show(true);

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
  pangolin::DataLog logEigR;
  pangolin::Plotter plotEigR(&logEigR, -100.f,1.f, -5.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotEigR);
  pangolin::DataLog logEigt;
  pangolin::Plotter plotEigt(&logEigt, -100.f,1.f, -5.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotEigt);
  gui.container().AddDisplay(plotters);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);

  tdp::ManagedHostImage<uint8_t> grey(wc, hc);
  tdp::ManagedHostImage<float> greyFl(wc,hc);
  tdp::ManagedDeviceImage<uint8_t> cuGrey(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreyFl(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyFlSmooth(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDu(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDv(wc,hc);
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

  tdp::ManagedHostImage<tdp::Vector3fda> pc_i;
  tdp::ManagedHostImage<tdp::Vector3bda> rgb_i;
  tdp::ManagedHostImage<tdp::Vector3fda> n_i;

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,10.);

  pangolin::Var<float> subsample("ui.subsample %",0.0005,0.0001,.001);

  pangolin::Var<float> scale("ui.scale",0.05,0.1,1);

  pangolin::Var<bool> useFAST("ui.use FAST",false,true);
  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updatePlanes("ui.update planes",true,true);
  pangolin::Var<bool> warmStartICP("ui.warmstart ICP",false,true);
  pangolin::Var<bool> useTexture("ui.use Tex in ICP",true,true);
  pangolin::Var<bool> incrementalAssign("ui.incremental assign ICP",false,true);
  pangolin::Var<float> lambdaTex("ui.ilamb Tex",0.1,0.0,1.);

  pangolin::Var<bool> icpReset("ui.reset icp",true,false);
  pangolin::Var<float> angleUniformityThr("ui.angle unif thr",5, 0, 90);
  pangolin::Var<float> angleThr("ui.angle Thr",15, -1, 90);
//  pangolin::Var<float> angleThr("ui.angle Thr",-1, -1, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.01,0,0.3);
  pangolin::Var<float> distThr("ui.dist Thr",0.1,0,0.3);
  pangolin::Var<float> HThr("ui.H Thr",-16.,-20,-10);
  pangolin::Var<float> negLogEvThr("ui.neg log ev Thr",-1,-2.,1.);
  pangolin::Var<float> condEntropyThr("ui.rel log dH ", 1.e-3,1.e-3,1e-2);
  pangolin::Var<int> maxIt("ui.max iter",30, 1, 20);

  pangolin::Var<int>   W("ui.W ",9,1,15);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool> showPlanes("ui.show planes",false,true);
  pangolin::Var<bool> showPcModel("ui.show model",false,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",false,true);
  pangolin::Var<bool> showFullPc("ui.show full",true,true);
  pangolin::Var<bool> showNormals("ui.show ns",false,true);
  pangolin::Var<bool> showAge("ui.show age",false,true);
  pangolin::Var<bool> showObs("ui.show # obs",false,true);
  pangolin::Var<bool> showNN("ui.show NN",false,true);
  pangolin::Var<int> step("ui.step",30,0,100);

  pangolin::Var<int> fastB("ui.FAST b",30,0,100);
  pangolin::Var<float> harrisThr("ui.harris thr",0.1,0.001,2.0);
  pangolin::Var<float> kappaHarris("ui.kappa harris",0.08,0.04,0.15);

  tdp::SE3f T_wc_0;
  tdp::SE3f T_wc = T_wc_0;
  std::vector<tdp::SE3f> T_wcs;
  Eigen::Matrix<float,6,6> Sigma_mc;
  std::vector<float> logHs;

  gui.verbose = true;
  if (gui.verbose) std::cout << "starting main loop" << std::endl;

  pangolin::GlBuffer vbo_w(pangolin::GlArrayBuffer,1000000,GL_FLOAT,3);
  pangolin::GlBuffer nbo_w(pangolin::GlArrayBuffer,1000000,GL_FLOAT,3);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(1000000);
  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  pangolin::GlBuffer cbo_w(pangolin::GlArrayBuffer,1000000,GL_UNSIGNED_BYTE,3);
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(1000000);
  rgb_w.Fill(tdp::Vector3bda::Zero());

  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_c(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_c(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_w(1000000);

  std::vector<std::pair<size_t, size_t>> mapNN;
  mapNN.reserve(10000000);

  std::mutex pl_wLock;
  std::thread mapping([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
    tdp::Plane pl;
    while(42) {
      pl.numObs_ = 0;
      {
        std::lock_guard<std::mutex> lock(pl_wLock); 
        if (pl_w.SizeToRead(iReadNext) > 0) {
          pl = pl_w.GetCircular(iReadNext);
        }
        iRead = pl_w.iRead_;
        iInsert = pl_w.iInsert_;
      }
      if (pl.numObs_ > 0) {
        float dotThr = cos(angleThr/180.*M_PI);
        while (iRead !=iInsert) {
          if (pl.Close(pl_w[iRead], dotThr, distThr, p2plThr))
            mapNN.emplace_back(iReadNext, iRead);
          iRead = (iRead+1)%pl_w.w_;
        }
        iReadNext = (iReadNext+1)%pl_w.w_;
      }
    };
  });

  tdp::ManagedHostImage<tdp::Brief> descs;
  tdp::ManagedHostImage<tdp::Vector2ida> pts;
  tdp::ManagedHostImage<float> orientation;

  tdp::ProjectiveAssociation<CameraT::NumParams, CameraT> projAssoc(cam, w, h);

  std::vector<std::pair<size_t, size_t>> assoc;
  assoc.reserve(10000);

  uint32_t numObs = 0;
  uint32_t numInlPrev = 0;

  tdp::DPvMFmeansSimple3f dpvmf(cos(50.*M_PI/180.));

  std::vector<std::vector<uint32_t>> invInd;
  std::vector<size_t> id_w;
  id_w.reserve(1000000);

  std::random_device rd;
  std::mt19937 gen(rd());

  mask.Fill(0);

  int32_t iReadCurW = 0;
  size_t frame = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (pangolin::Pushed(icpReset)) {
      T_wc = tdp::SE3f();
    }

    if (!gui.paused() 
        && frame > 0
        && (runMapping || frame == 1) 
        && (trackingGood || frame < 10)) { // add new observations
      TICK("mask");
//      tdp::RandomMaskCpu(mask, perc, W*dMax);
//      tdp::UniformResampleMask(mask, W, subsample, gen, 4, 4);
      if (useFAST) {
        tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, W, pts,
            orientation);
        mask.Fill(0);
        for (size_t i=0; i<pts.Area(); ++i) {
          mask(pts[i](0), pts[i](1)) = 1;
        }
//        tdp::ExtractBrief(grey, pts, orientations, gui.frame, descs);
      } else {
//        tdp::UniformResampleEmptyPartsOfMask(mask, W, subsample, gen, 16, 16);
//        tdp::UniformResampleMask(pc, cam, mask, W, subsample, gen, 16, 16);
        tdp::UniformResampleEmptyPartsOfMask(pc, cam, mask, W,
            subsample, gen, 16, 16, w, h);
      }
      TOCK("mask");
      {
        iReadCurW = pl_w.iInsert_;
        std::lock_guard<std::mutex> lock(pl_wLock); 
        TICK("normals");
        ExtractNormals(pc, rgb, grey, greyFl, gradGrey, mask, W, frame,
            T_wc, pl_w, pc_w,
            rgb_w, n_w);
        TOCK("normals");

        TICK("add to model");
        for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
          tdp::Plane& pl = pl_w[i];
          dpvmf.addObservation(pl.n_);
          uint32_t zi = dpvmf.GetZs().back();
          int32_t kMax = -1;
          uint32_t nMax = 0;
          for (size_t k=0; k<dpvmf.GetK(); ++k) {
            if (k==zi) continue;
            if (nMax < dpvmf.GetNs()[k]) {
              nMax = dpvmf.GetNs()[k];
              kMax = k;
            }
          }
          pl.dir_ = dpvmf.GetCenter(kMax);
        }
      }

//      vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
      vbo_w.Upload(&pc_w.ptr_[iReadCurW], 
          pc_w.SizeToRead(iReadCurW)*sizeof(tdp::Vector3fda), 
          iReadCurW*sizeof(tdp::Vector3fda));

      id_w.resize(pl_w.SizeToRead());
      std::iota(id_w.begin(), id_w.end(), 0);
      std::random_shuffle(id_w.begin(), id_w.end());
      TOCK("add to model");
      std::cout << " # map points: " << pl_w.SizeToRead() 
        << " " << dpvmf.GetZs().size() << std::endl;
      TICK("dpvmf");
      dpvmf.iterateToConvergence(100, 1e-6);
      for (size_t k=0; k<dpvmf.GetK(); ++k) {
        if (k >= invInd.size()) {
          invInd.push_back(std::vector<uint32_t>());
          invInd.back().reserve(10000);
        } else {
          invInd[k].clear();
        }

        if (incrementalAssign) {
          for (auto i : id_w) {
            if (dpvmf.GetZs()[i] == k) 
              invInd[k].push_back(i);
            if (invInd[k].size() >= 10000)
              break;
          }
        }
//        std::cout << "cluster " << k << ": # " << invInd[k].size() 
//          << " of " << dpvmf.GetNs()[k] << std::endl;
//        std::sort(invInd[k].begin(), invInd[k].begin(), 
//            [&](uint32_t a, uint32_t b) {
//            return pl_w[a].numObs_ > pl_w[b].numObs_;
//            });
//        std::cout << pl_w[invInd[k][0]].numObs_ 
//          << " " << pl_w[invInd[k][1]].numObs_ << std::endl;
      }
      TOCK("dpvmf");
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();

    int64_t t_host_us_d = 0;
    TICK(
        "Setup");
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
    tdp::Blur5(cuGreyFl,cuGreyFlSmooth, 10.);
    tdp::Convert(cuGreyFlSmooth, cuGrey, 255.);
    grey.CopyFrom(cuGrey);
    greyFl.CopyFrom(cuGreyFlSmooth);
    tdp::Gradient(cuGreyFlSmooth, cuGreyDu, cuGreyDv, cuGradGrey);

    n.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    size_t numProjected =0;
    trackingGood = false;
    if (frame > 1 && runTracking) { // tracking
      TICK("icp");
      Eigen::Matrix<float,6,6> A;
      Eigen::Matrix<float,6,1> b;
      Eigen::Matrix<float,6,1> Ai;
      float dotThr = cos(angleThr*M_PI/180.);

      std::uniform_int_distribution<> dis(0, dpvmf.GetK()-1);
      
      std::vector<size_t> indK(dpvmf.GetK(),0);
      for (size_t it = 0; it < maxIt; ++it) {
        if (it % 1 == 0) {
          mask.Fill(0);
          assoc.clear();
          pc_c.MarkRead();
          n_c.MarkRead();
          indK = std::vector<size_t>(dpvmf.GetK(),0);
          numProjected = 0;
        }

        A = Eigen::Matrix<float,6,6>::Zero();
        b = Eigen::Matrix<float,6,1>::Zero();
        Ai = Eigen::Matrix<float,6,1>::Zero();
        float err = 0.;
        float H = 1e10;
        float Hprev = 1e10;
        uint32_t numInl = 0;
        numObs = 0;

        tdp::SE3f T_cw = T_wc.Inverse();
        if (warmStartICP) {
          Eigen::Matrix3f N = Eigen::Matrix3f::Zero();
          bool exploredAll = false;
          uint32_t k = dis(gen);
          while (numObs < dpvmf.GetK()*10 && !exploredAll) {
            k = (k+1) % dpvmf.GetK();
            while (indK[k] < invInd[k].size()) {
              size_t i = invInd[k][indK[k]++];
              tdp::Plane& pl = pl_w.GetCircular(i);
              numProjected++;
              int32_t u, v;
              if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, cam, pc,
                    W, n, u,v ))
                continue;
              if (AccumulateRot(pl, T_wc, T_cw, pc(u,v), n(u,v),
                    distThr, p2plThr, dotThr, N)) {
                pl.lastFrame_ = frame;
                pl.numObs_ ++;
                numInl ++;
                mask(u,v) ++;
                assoc.emplace_back(i,pc_c.SizeToRead());
                pc_c.Insert(pc(u,v));
                n_c.Insert(n(u,v));
                break;
              }
            }
            exploredAll = true;
            for (size_t k=0; k<indK.size(); ++k) 
              exploredAll &= indK[k] >= invInd[k].size();
          }
          Eigen::JacobiSVD<Eigen::Matrix3d> svd(N.cast<double>(),
              Eigen::ComputeFullU | Eigen::ComputeFullV);
          Eigen::Matrix3f R_wc = (svd.matrixU()*svd.matrixV().transpose()).cast<float>();
          T_wc.rotation() = tdp::SO3f(R_wc);

          // first use already associated data
          for (const auto& ass : assoc) {
            tdp::Plane& pl = pl_w.GetCircular(ass.first);
            if (AccumulateP2Pl(pl, T_wc, T_cw,
                  pc_c.GetCircular(ass.second), 
                  n_c.GetCircular(ass.second),
                  distThr, p2plThr, dotThr, A, Ai, b, err)) {
              if (tdp::CheckEntropyTermination(A, Hprev, HThr,
                    condEntropyThr, negLogEvThr, H))
                break;
              Hprev = H;
              numObs ++;
            }
          }
          std::cout << " reused " << numInl << " of " << assoc.size() << std::endl;
        }
//        size_t numInl0 = numInl;
        numInlPrev = numInl;
        // associate new data until enough
        bool exploredAll = false;
        uint32_t k = dis(gen);
        if (incrementalAssign) {
          while (numObs < 10000 && !exploredAll) {
            k = (k+1) % dpvmf.GetK();
            while (indK[k] < invInd[k].size()) {
              size_t i = invInd[k][indK[k]++];
              tdp::Plane& pl = pl_w.GetCircular(i);
              numProjected++;
              int32_t u, v;
              if (angleThr > 0.) {
                if (!tdp::ProjectiveAssocNormalExtract(pl, T_cw, cam, pc,
                      W, n, u,v ))
                  continue;
                if (useTexture) {
                  if (!AccumulateP2Pl(pl, T_wc, T_cw, cam, pc(u,v), n(u,v), 
                        grey(u,v), distThr, p2plThr, dotThr, lambdaTex,
                        A, Ai, b, err))
                    continue;
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
              assoc.emplace_back(i,pc_c.SizeToRead());
              pc_c.Insert(pc(u,v));
              n_c.Insert(n(u,v));
              break;
            }

            if (numInl > numInlPrev) {
              if (tdp::CheckEntropyTermination(A, Hprev, HThr, condEntropyThr, 
                    negLogEvThr, H))
                break;
              Hprev = H;
              numObs ++;
            }
            numInlPrev = numInl;

            exploredAll = true;
            for (size_t k=0; k<indK.size(); ++k) {
              exploredAll &= indK[k] >= invInd[k].size();
            }
          }
        } else {
          TICK("data assoc");
          projAssoc.Associate(vbo_w, T_wc.Inverse(), dMin, dMax, 
              pl_w.SizeToRead());
          TOCK("data assoc");
          TICK("extract assoc");
          z.Fill(0);
          projAssoc.GetAssoc(z, mask);
          // reset and reconstruct inverted index
          std::vector<std::vector<uint32_t>> invUV;
          for (size_t k=0; k<invInd.size(); ++k) {
            invInd[k].clear();
            invUV.push_back(std::vector<uint32_t>());
          }
          for (size_t i=0; i<z.Area(); ++i)
            if (z[i] > 0) {
              invInd[dpvmf.GetZs()[z[i]-1]].push_back(z[i]-1);
              invUV[dpvmf.GetZs()[z[i]-1]].push_back(i);
            }
          TOCK("extract assoc");
//          for (size_t k=0; k<invInd.size(); ++k) {
//            std::cout << invInd[k].size() 
//              << " " << invUV[k].size() << std::endl;
//          }
          TICK("accumulate");
          float tN = 0, numN = 0;
          float tAcc = 0, numAcc = 0;
          float tEnt = 0, numEnt = 0;
          tdp::Timer t0;
          while (numObs < 10000 && !exploredAll) {
            k = (k+1) % dpvmf.GetK();
            while (indK[k] < invInd[k].size()) {
              int32_t u = invUV[k][indK[k]] % z.w_;
              int32_t v = invUV[k][indK[k]] / z.w_;
              size_t i = invInd[k][indK[k]++];
              tdp::Plane& pl = pl_w.GetCircular(i);
              numProjected++;
              t0.tic();
              if (!EnsureNormal(pc, W, n, u, v)) {
                tN += t0.toc(); numN ++;
                continue;
              }
              tN += t0.toc(); numN ++;
              if (useTexture) {

                t0.tic();
                if (!AccumulateP2Pl(pl, T_wc, T_cw, cam, pc(u,v), n(u,v), 
                      grey(u,v), distThr, p2plThr, dotThr, lambdaTex,
                      A, Ai, b, err)) {
                  tAcc  += t0.toc(); numAcc ++;
                  continue;
                }
                tAcc  += t0.toc(); numAcc ++;
              } else {
                if (!AccumulateP2Pl(pl, T_wc, T_cw, pc(u,v), n(u,v),
                      distThr, p2plThr, dotThr, A, Ai, b, err))
                  continue;
              }
              pl.lastFrame_ = frame;
              pl.numObs_ ++;
              numInl ++;
              assoc.emplace_back(i,pc_c.SizeToRead());
              pc_c.Insert(pc(u,v));
              n_c.Insert(n(u,v));
              break;
            }
            if (numInl > numInlPrev) {
              t0.tic();
              if (tdp::CheckEntropyTermination(A, Hprev, HThr, condEntropyThr, 
                    negLogEvThr, H)) {
                tEnt  += t0.toc(); numEnt ++;
                break;
              }
              tEnt  += t0.toc(); numEnt ++;
              Hprev = H;
              numObs ++;
            }
            numInlPrev = numInl;
            exploredAll = true;
            for (size_t k=0; k<indK.size(); ++k) {
              exploredAll &= indK[k] >= invInd[k].size();
            }
          }
          TOCK("accumulate");
          std::cout 
            << "ICP timings: normals " << tN/numN << " " << tN 
            << " accumulation " << tAcc/numAcc << " " << tAcc 
            << " entropy " << tEnt/numEnt << " " << tEnt << std::endl;
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
        if (x.norm() < 1e-4
            || tdp::CheckEntropyTermination(A, Hprev, HThr, 
               0.f, negLogEvThr, H)) {
          std::cout << numInl << " " << numObs << " " << numProjected << std::endl;
          break;
        }
      }
      for (size_t k=0; k<indK.size(); ++k) {
        std::cout << "used different directions: " << indK[k] 
          << " of " << invInd[k].size() << std::endl;
      }
      Sigma_mc = A.inverse();
      logObs.Log(log(numObs)/log(10.), log(numInlPrev)/log(10.), 
          log(numProjected)/log(10.), log(pl_w.SizeToRead())/log(10));
      Eigen::Matrix<float,6,1> ev = Sigma_mc.eigenvalues().real();
      float H = ev.array().log().sum();
      std::cout << " H " << H << " neg log evs " 
        << ev.array().log().matrix().transpose()
        << std::endl;

//      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
//      std::cout << eig.eigenvalues().real().transpose() << std::endl;
//      Eigen::Matrix<float,6,6> Q = eig.eigenvectors();
//      for (size_t k=0; k<dpvmf.GetK(); ++k) {
//        Eigen::Matrix<float,6,1> Ai;
//        Ai << Eigen::Vector3f::Zero(), dpvmf.GetCenter(k);
//        std::cout << "k " << k << std::endl;
//        std::cout << (Q.transpose()*Ai*Ai.transpose()*Q).diagonal().transpose() << std::endl;
//      }

      logEntropy.Log(H);
      logEigR.Log(ev.topRows<3>().array().log().matrix());
      logEigt.Log(ev.bottomRows<3>().array().log().matrix());
      T_wcs.push_back(T_wc);
      trackingGood = H <= HThr && numInlPrev > 10;
      TOCK("icp");
      if (trackingGood) {
        std::cout << "tracking good" << std::endl;
      }

      if (updatePlanes && trackingGood) {
        TICK("update planes");
        for (const auto& ass : assoc) {
          tdp::Vector3fda pc_c_in_w = T_wc*pc_c.GetCircular(ass.second);
          tdp::Vector3fda n_c_in_w = T_wc.rotation()*n_c.GetCircular(ass.second);
          pl_w.GetCircular(ass.first).AddObs(pc_c_in_w, n_c_in_w);
          n_w.GetCircular(ass.first) = pl_w.GetCircular(ass.first).n_;
          pc_w.GetCircular(ass.first) = pl_w.GetCircular(ass.first).p_;
        }
        TOCK("update planes");
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

      pangolin::glDrawAxis(T_wc.matrix(), 0.05f);
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,20);

      if (showFullPc) {
        vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
        cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeBytes(), 0);
        if ((!showAge && !showObs) || pl_w.SizeToRead() == 0) {
          pangolin::RenderVboCbo(vbo_w, cbo_w, true);
        } else {
          age.Reinitialise(pl_w.SizeToRead());
          if (showAge) {
            for (size_t i=0; i<age.Area(); ++i) 
              age[i] = pl_w.GetCircular(i).lastFrame_;
          } else {
            for (size_t i=0; i<age.Area(); ++i) 
              age[i] = pl_w.GetCircular(i).numObs_;
          }
          valuebo.Reinitialise(pangolin::GlArrayBuffer, age.Area(),  GL_FLOAT,
              1, GL_DYNAMIC_DRAW);
          valuebo.Upload(age.ptr_,  age.SizeBytes(), 0);

          pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
          pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
          std::pair<float,float> minMaxAge = age.MinMax();
//          std::cout << " age " << minMaxAge.first 
//            << " < . < " << minMaxAge.second << std::endl;
          tdp::RenderVboValuebo(vbo_w, valuebo, minMaxAge.first, minMaxAge.second,
              P, MV);
        }
        if (showNN) {
          glColor4f(0.3,0.3,0.3,0.3);
          for (auto& ass : mapNN) {
            tdp::glDrawLine(pl_w[ass.first].p_, pl_w[ass.second].p_);
          }
        }
      }

      if (showNormals) {
        glColor4f(1,0,0.,0.5);
        pangolin::glSetFrameOfReference(T_wc.matrix());
        for (size_t i=0; i<n_i.Area(); ++i) {
          tdp::glDrawLine(pc_i[i], pc_i[i] + scale*n_i[i]);
        }
        for (size_t i=0; i<n_c.SizeToRead(); ++i) {
          tdp::glDrawLine(pc_c.GetCircular(i), 
              pc_c.GetCircular(i) + scale*n_c.GetCircular(i));
        }
        pangolin::glUnsetFrameOfReference();
        glColor4f(0,1,0,0.5);
        for (size_t i=0; i<n_w.SizeToRead(); i+=step) {
          tdp::glDrawLine(pc_w.GetCircular(i), 
              pc_w.GetCircular(i) + scale*n_w.GetCircular(i));
        }
      }

      if (showPlanes) {
        for (size_t i=iReadCurW; i != pl_w.iInsert_; i=(i+1)%pl_w.w_) {
          tdp::SE3f T = pl_w.GetCircular(i).LocalCosy();
          pangolin::glDrawAxis(T.matrix(),0.05f);
        }
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
        tdp::Vector3fda pc_c_in_m = T_wc*pc_c.GetCircular(ass.second);
        tdp::glDrawLine(pl_w.GetCircular(ass.first).p_, pc_c_in_m);
      }
    }

    if (viewNormals.IsShown()) {
      viewNormals.Activate(s_cam);
      glColor4f(0,0,1,0.5);
      nbo_w.Upload(n_w.ptr_, n_w.SizeBytes(), 0);
      pangolin::RenderVbo(nbo_w);
      glColor4f(1,0,0,1.);
      for (size_t k=0; k<dpvmf.GetK(); ++k) {
        tdp::glDrawLine(tdp::Vector3fda::Zero(), dpvmf.GetCenter(k));
      }
      glColor4f(0,1,0,1.);
      tdp::SE3f R_wc(T_wc.rotation());
      pangolin::glSetFrameOfReference(R_wc.matrix());
      vbo.Reinitialise(pangolin::GlArrayBuffer, n_i.Area(), GL_FLOAT,
          3, GL_DYNAMIC_DRAW);
      vbo.Upload(n_i.ptr_, n_i.SizeBytes(), 0);
      pangolin::RenderVbo(vbo);
      pangolin::glUnsetFrameOfReference();
    }

    TOCK("Draw 3D");
    if (gui.verbose) std::cout << "draw 2D" << std::endl;
    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (containerTracking.IsShown()) {
      if (viewCurrent.IsShown()) {
        std::cout << "set rgb image" << std::endl;
        viewCurrent.SetImage(rgb);
        std::cout << "done set rgb image" << std::endl;
      }
      if (viewMask.IsShown()) {
        std::cout << "set mask image" << std::endl;
        viewMask.SetImage(mask);
      }
    }
    std::cout << "scroll plots" << std::endl;
    plotdH.ScrollView(1,0);
    plotH.ScrollView(1,0);
    plotObs.ScrollView(1,0);
    plotEigR.ScrollView(1,0);
    plotEigt.ScrollView(1,0);

    TOCK("Draw 2D");

    if (gui.verbose) std::cout << "finished one iteration" << std::endl;
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }

  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;
  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

