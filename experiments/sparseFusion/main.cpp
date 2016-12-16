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

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

namespace tdp {

//void ExtractNormals(const Image<Vector3fda>& pc, 
//    const Image<Vector3bda>& rgb,
//    const Image<uint8_t>& mask, uint32_t W,
//    ManagedHostImage<Vector3fda>& pc_c,
//    ManagedHostImage<Vector3bda>& rgb_c,
//    ManagedHostImage<Vector3fda>& n_c) {
//  size_t numObs = 0;
//  for (size_t i=0; i<mask.Area(); ++i) {
//    if (mask[i] && tdp::IsValidData(pc[i])
//        && pc[i].norm() < 5. 
//        && 0.3 < pc[i].norm() ) 
//      numObs++;
//  }
//  pc_c.Reinitialise(numObs);
//  rgb_c.Reinitialise(numObs);
//  n_c.Reinitialise(numObs);
//  size_t j=0;
//  for (size_t i=0; i<mask.Area(); ++i) {
//    if (mask[i] && tdp::IsValidData(pc[i])
//        && pc[i].norm() < 5. 
//        && 0.3 < pc[i].norm() )  {
//      uint32_t u0 = i%mask.w_;
//      uint32_t v0 = i/mask.w_;
//      pc_c[j] = pc(u0,v0);
//      rgb_c[j] = rgb(u0,v0);
//      uint32_t Wscaled = floor(W*pc_c[j](2));
//      if (!tdp::NormalViaScatter(pc, u0, v0, Wscaled, n_c[j++])) {
//        std::cout << "problem with normal computation" << std::endl;
//        std::cout << u0 << " " << v0 << std::endl;
//        std::cout << pc_c[j].transpose() << std::endl;
//      }
//    }
//  }
//}

void ExtractNormals(const Image<Vector3fda>& pc, 
    const Image<Vector3bda>& rgb,
    const Image<uint8_t>& mask, uint32_t W,
    ManagedHostImage<Vector3fda>& pc_c,
    ManagedHostImage<Vector3bda>& rgb_c,
    ManagedHostImage<Vector3fda>& n_c) {
  std::vector<size_t> ids;
  eigen_vector<Vector3fda> ns;
  ids.reserve(1000);
  ns.reserve(1000);
  Vector3fda n;
  for (size_t i=0; i<mask.Area(); ++i) {
    if (mask[i] && tdp::IsValidData(pc[i])
        && pc[i].norm() < 5. 
        && 0.3 < pc[i].norm() )  {
      uint32_t Wscaled = floor(W*pc[i](2));
      if (tdp::NormalViaScatter(pc, i%mask.w_, i/mask.w_, Wscaled, n)) {
        ids.push_back(i);
        ns.emplace_back(n);
      }
    }
  }
  size_t numObs = ids.size();
  pc_c.Reinitialise(numObs);
  rgb_c.Reinitialise(numObs);
  n_c.Reinitialise(numObs);
  for (size_t j=0; j<ids.size(); ++j) {
    size_t i= ids[j];
    pc_c[j] = pc[i];
    rgb_c[j] = rgb[i];
    n_c[j] = ns[j];
  }
}

struct Plane {
  Vector3fda p_; 
  Vector3fda n_; 
  Vector3bda rgb_; 

  uint32_t lastFrame_;
  uint32_t numObs_;

  float w_; // weight

  void AddObs(const Vector3fda& p, const Vector3fda& n) {
    float wNew = w_+1; 
    p_ = (p_*w_ + p)/wNew;
    n_ = (n_*w_ + n).normalized();
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
};

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

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewModel(wc, hc);
  containerTracking.AddDisplay(viewModel);
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
  pangolin::Plotter plotObs(&logObs, -100.f,1.f, 0.0f,10000.f, .1f, 0.1f);
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
  pangolin::DataLog logAdaptiveEntropy;
  pangolin::Plotter plotAdaptiveH(&logAdaptiveEntropy, -100.f,1.f,
      -40.f,40.f, .1f, 0.1f);
  plotters.AddDisplay(plotAdaptiveH);
  gui.container().AddDisplay(plotters);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb_m(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);

  tdp::ManagedHostImage<tdp::Vector3fda> pcFull_m(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);
  tdp::ManagedDeviceImage<float> cuGrey(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedDeviceImage<uint8_t> cuMask(wc, hc);
  tdp::ManagedHostImage<uint8_t> mask(wc, hc);

  // ICP stuff
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

//  tdp::ManagedHostImage<tdp::Vector3fda> pc_c;
//  tdp::ManagedHostImage<tdp::Vector3bda> rgb_c;
//  tdp::ManagedHostImage<tdp::Vector3fda> n_c;

  tdp::ManagedHostImage<tdp::Vector3fda> pc_i;
  tdp::ManagedHostImage<tdp::Vector3bda> rgb_i;
  tdp::ManagedHostImage<tdp::Vector3fda> n_i;

  tdp::ManagedHostImage<tdp::Vector3fda> pc_m;
  tdp::ManagedHostImage<tdp::Vector3fda> n_m;

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,10.);

  pangolin::Var<float> subsample("ui.subsample %",0.001,0.0001,.01);

  pangolin::Var<float> scale("ui.scale %",0.1,0.1,1);

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updatePlanes("ui.update planes",false,true);

  pangolin::Var<bool> icpReset("ui.reset icp",true,false);
  pangolin::Var<float> angleUniformityThr("ui.angle unif thr",5, 0, 90);
  pangolin::Var<float> angleThr("ui.angle Thr",15, 0, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.01,0,0.3);
  pangolin::Var<float> distThr("ui.dist Thr",0.1,0,0.3);
  pangolin::Var<float> HThr("ui.H Thr",-16.,-20,-10);
  pangolin::Var<float> relLogHChange("ui.rel log dH ", 1.e-2,1.e-3,1e-2);
  pangolin::Var<int> maxIt("ui.max iter",20, 1, 20);

  pangolin::Var<int>   W("ui.W ",8,1,15);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool> showPlanes("ui.show planes",false,true);
  pangolin::Var<bool> showPcModel("ui.show model",false,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",false,true);
  pangolin::Var<bool> showFullPc("ui.show full",true,true);
  pangolin::Var<bool> showNormals("ui.show ns",false,true);

  tdp::SE3f T_wc_0;
  tdp::SE3f T_wc = T_wc_0;
  std::vector<tdp::SE3f> T_wcs;
  Eigen::Matrix<float,6,6> Sigma_mc;
  std::vector<float> logHs;

  gui.verbose = true;
  if (gui.verbose) std::cout << "starting main loop" << std::endl;

  pangolin::GlBuffer vbo_w(pangolin::GlArrayBuffer,1000000,GL_FLOAT,3);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(1000000);
  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  pangolin::GlBuffer cbo_w(pangolin::GlArrayBuffer,1000000,GL_UNSIGNED_BYTE,3);
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(1000000);
  rgb_w.Fill(tdp::Vector3bda::Zero());

  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_c(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_c(1000000);

  std::vector<std::pair<size_t, size_t>> assoc;
  assoc.reserve(10000);

  uint32_t numObs = 0;
  uint32_t numInlPrev = 0;

  tdp::DPvMFmeansSimple3f dpvmf(cos(65.*M_PI/180.));

  std::vector<std::vector<uint32_t>> invInd;
  std::vector<size_t> id_w;
  id_w.reserve(1000000);

  std::random_device rd;
  std::mt19937 gen(rd());

  mask.Fill(0);

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
        && (trackingGood || frame == 1)) { // add new observations
      TICK("mask");
//      tdp::RandomMaskCpu(mask, perc, W*dMax);
      std::uniform_real_distribution<> coin(0, 1);
      size_t I = 4;
      size_t J = 4;
      for (size_t i=0; i<I; ++i) {
        for (size_t j=0; j<J; ++j) {
          size_t count = 0;
          for (size_t u=i*mask.w_/I; u<(i+1)*mask.w_/I; ++u) {
            for (size_t v=j*mask.h_/J; v<(j+1)*mask.h_/J; ++v) {
              if (mask(u,v)) count++;
            }
          }
          float perc = (float)subsample-(float)count/(float)(mask.w_/I*mask.h_/J);
          std::cout << i << "," << j << ": " << 100*perc << std::endl;
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
      TOCK("mask");
      TICK("normals");
      ExtractNormals(pc, rgb, mask, W, pc_i, rgb_i, n_i);
      TOCK("normals");

      TICK("add to model");
      tdp::Plane pl;
      for (size_t i=0; i<pc_i.Area(); ++i) {
        pl.p_ = T_wc*pc_i[i];
        pl.n_ = T_wc.rotation()*n_i[i];
        pl.rgb_ = rgb_i[i];
        pl.lastFrame_ = frame;
        pl.numObs_ = 1;
        pl_w.Insert(pl);
        pc_w.Insert(pl.p_);
        rgb_w.Insert(pl.rgb_);

        dpvmf.addObservation(pl.n_);
      }
      id_w.resize(pl_w.SizeToRead());
      std::iota(id_w.begin(), id_w.end(), 0);
      std::random_shuffle(id_w.begin(), id_w.end());
      TOCK("add to model");
      std::cout << " # map points: " << pl_w.SizeToRead() 
        << " " << dpvmf.GetZs().size() << std::endl;
      TICK("dpvmf");
      dpvmf.iterateToConvergence(100, 1e-3);
      for (size_t k=0; k<dpvmf.GetK(); ++k) {
        if (k >= invInd.size()) {
          invInd.push_back(std::vector<uint32_t>());
          invInd.back().reserve(1000000);
        } else {
          invInd[k].clear();
        }
        for (auto i : id_w) {
          if (dpvmf.GetZs()[i] == k) 
            invInd[k].push_back(i);
          if (invInd[k].size() >= 10000)
            break;
        }
        std::cout << "cluster " << k << ": # " << invInd[k].size() << std::endl;
      }
      TOCK("dpvmf");
      std::cout << " # clusters " << dpvmf.GetK() << " " 
        << dpvmf.GetNs().size() << std::endl;

      if (showFullPc) {
        vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
        cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeBytes(), 0);
      }
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

      if (viewMask.IsShown()) {
        viewMask.SetImage(mask);
      }

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
    n.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    pc_c.MarkRead();

    size_t numProjected =0;
    trackingGood = false;
    if (frame > 1 && runTracking) { // tracking
      TICK("icp");
      Eigen::Matrix<float,6,6> A;
      Eigen::Matrix<float,6,1> b;
      Eigen::Matrix<float,6,1> Ai;
      float dotThr = cos(angleThr*M_PI/180.);

//      TICK("icp prep");
//      TOCK("icp prep");

      std::uniform_int_distribution<> dis(0, dpvmf.GetK()-1);
      
      mask.Fill(0);
      for (size_t it = 0; it < maxIt; ++it) {
        assoc.clear();

        A = Eigen::Matrix<float,6,6>::Zero();
        b = Eigen::Matrix<float,6,1>::Zero();
        Ai = Eigen::Matrix<float,6,1>::Zero();
        float bi = 0.;
        float err = 0.;
        float Hprev = 1e10;
        uint32_t numInl = 0;
        numObs = 0;

        std::vector<size_t> indK(dpvmf.GetK(),0);
//        tdp::eigen_vector<tdp::Vector3fda> ns;

        tdp::SE3f T_cw = T_wc.Inverse();
        bool exploredAll = false;
        uint32_t k = dis(gen);
        while (numObs < 10000 && !exploredAll) {
          k = (k+1) % dpvmf.GetK();
          while (indK[k] < invInd[k].size()) {
//            std::cout << k << " " << indK[k] << " " << invInd[k].size() << std::endl;
            size_t i = invInd[k][indK[k]++];

            tdp::Vector3fda& n_w = pl_w.GetCircular(i).n_;
            tdp::Vector3fda& pc_w = pl_w.GetCircular(i).p_;

            tdp::Vector3fda pc_w_in_c = T_cw*pc_w;
            Eigen::Vector2f x = cam.Project(pc_w_in_c);
            int32_t u = floor(x(0)+0.5f);
            int32_t v = floor(x(1)+0.5f);
            numProjected++;
            if (0 <= u && u < w && 0 <= v && v < h) {
              if (tdp::IsValidData(pc(u,v))) {
                uint32_t Wscaled = floor(W*pc(u,v)(2));
                tdp::Vector3fda ni = n(u,v);
                if (!tdp::IsValidData(ni)) {
                  if(!tdp::NormalViaScatter(pc, u, v, Wscaled, ni)) {
                    continue;
                  } else {
                    n(u,v) = ni;
                  }
                }

                tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);
                float dist = (pc_w - pc_c_in_w).norm();
                if (dist < distThr*pc(u,v)(2)) {
                  Eigen::Vector3f n_w_in_c = T_cw.rotation()*n_w;
                  if (fabs(n_w_in_c.dot(ni)) > dotThr) {
                    float p2pl = n_w.dot(pc_w - pc_c_in_w);
                    if (fabs(p2pl) < p2plThr) {
                      Ai.topRows<3>() = pc(u,v).cross(n_w_in_c); 
                      Ai.bottomRows<3>() = n_w_in_c; 
                      bi = p2pl;
                      A += Ai * Ai.transpose();
                      b += Ai * bi;
                      err += p2pl;
                      assoc.emplace_back(i,pc_c.SizeToRead());
                      pc_c.Insert(pc(u,v));
                      n_c.Insert(ni);
                      // if this gets expenseive I could use 
                      // https://en.wikipedia.org/wiki/Matrix_determinant_lemma
                      numInl ++;
                      pl_w.GetCircular(i).lastFrame_ = frame;
                      pl_w.GetCircular(i).numObs_ ++;
                      mask(u,v) ++;

                      break;
                    }
                  }
                }
              }
            }
          }

          if (numInl > numInlPrev) {
            float H = -((A.eigenvalues()).array().log().sum()).real();
//            std::cout << numObs << " " << numInl 
//              << " H " << H << " delta "
//              << (Hprev-H) << std::endl;
            if ( (H < HThr ||  (Hprev - H) < relLogHChange )
                && numInl > 6) {
              std::cout << numInl 
                << " " << numObs
                << " " << numProjected 
                << " H " << H 
                << " delta " << (Hprev-H) 
                << " " << -A.eigenvalues().array().log().matrix().transpose()
                << std::endl;
              for (auto k : indK) std::cout << k << " " ;
              std::cout << std::endl;
              break;
            }
            logAdaptiveEntropy.Log(H);
            Hprev = H;
            numObs ++;
          }
          numInlPrev = numInl;

          exploredAll = true;
          for (size_t k=0; k<indK.size(); ++k) 
            exploredAll &= indK[k] >= invInd[k].size();
//          for (auto k : indK) std::cout << k << " " ;
//          std::cout << std::endl;
        }
        Eigen::Matrix<float,6,1> x = Eigen::Matrix<float,6,1>::Zero();
        if (numInl > 10) {
          // solve for x using ldlt
          x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
          T_wc = T_wc * tdp::SE3f::Exp_(x);
        }
        if (gui.verbose) {
          std::cout << " it " << it << ": err=" << err 
            << "\t# inliers: " << numInl
            << "\t|x|: " << x.topRows(3).norm()*180./M_PI 
            << " " <<  x.bottomRows(3).norm()
            << std::endl;
        }
        if (x.norm() < 1e-4) break;
      }
      Sigma_mc = A.inverse();
      logObs.Log(numObs, numInlPrev, numProjected);
      plotAdaptiveH.ScrollView(numObs,0);
      Eigen::Matrix<float,6,1> ev = Sigma_mc.eigenvalues().real();
      float H = ev.array().log().sum();
      std::cout << " H " << H  << std::endl;
      logEntropy.Log(H);
      logEigR.Log(0.5*ev.topRows<3>().array().log().matrix());
      logEigt.Log(0.5*ev.bottomRows<3>().array().log().matrix());
      T_wcs.push_back(T_wc);
      trackingGood = true;
      TOCK("icp");

      if (updatePlanes) {
        TICK("update planes");
        for (const auto& ass : assoc) {
          tdp::Vector3fda pc_c_in_w = T_wc*pc_c.GetCircular(ass.second);
          tdp::Vector3fda n_c_in_w = T_wc.rotation()*n_c[ass.second];
          pl_w.GetCircular(ass.first).AddObs(pc_c_in_w, n_c_in_w);
        }
        TOCK("update planes");
      }
    }

    frame ++;

    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    TICK("Draw 3D");
    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      pangolin::glDrawAxis(T_wc.matrix(), 0.05f);
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,20);

      if (showFullPc) {
//        glColor4f(0.,1.,1.,0.6);
//        pangolin::RenderVbo(vbo_w);
        pangolin::RenderVboCbo(vbo_w, cbo_w, true);
      }

      glColor3f(1,0,0);
      if (showNormals) {
        for (size_t i=0; i<n_m.Area(); ++i) {
          tdp::glDrawLine(pc_m[i], pc_m[i] + scale*n_m[i]);
        }
      } else {
//        vbo.Reinitialise(pangolin::GlArrayBuffer, pc_m.Area(), GL_FLOAT,
//            3, GL_DYNAMIC_DRAW);
//        vbo.Upload(pc_m.ptr_, pc_m.SizeBytes(), 0);
//        pangolin::RenderVbo(vbo);
      }

      glColor3f(0,1,0);
      pangolin::glSetFrameOfReference(T_wc.matrix());
//      if (showNormals) {
//        for (size_t i=0; i<n_c.Area(); ++i) {
//          tdp::glDrawLine(pc_c[i], pc_c[i] + scale*n_c[i]);
//        }
//      } else {
//        vbo.Reinitialise(pangolin::GlArrayBuffer, pc_c.Area(), GL_FLOAT,
//            3, GL_DYNAMIC_DRAW);
//        vbo.Upload(pc_c.ptr_, pc_c.SizeBytes(), 0);
//        pangolin::RenderVbo(vbo);
//      }
      pangolin::glUnsetFrameOfReference();

      if (showPlanes) {
        for (size_t i=0; i<n_c.Area(); ++i) {
          Eigen::Matrix3f R = tdp::OrthonormalizeFromYZ(
              Eigen::Vector3f(0,1,0), n_c[i].normalized());
          tdp::SE3f T(R, pc_c[i]); 
          pangolin::glSetFrameOfReference((T_wc*T).matrix());
          pangolin::glDrawAxis(0.05f);
          pangolin::glDraw_z0(0.01,10);
          pangolin::glUnsetFrameOfReference();
        }
      }

      // render current camera second in the propper frame of
      // reference
      if (showPcCurrent) {
        vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(), GL_FLOAT,
            3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
        pangolin::glSetFrameOfReference(T_wc.matrix());
        if(dispLvl == 0){
          cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
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

      pangolin::glSetFrameOfReference((T_wc).matrix());
      pangolin::glDrawAxis(0.1f);
//      vbo.Reinitialise(pangolin::GlArrayBuffer, pc_c.Area(), GL_FLOAT,
//          3, GL_DYNAMIC_DRAW);
//      vbo.Upload(pc_c.ptr_, pc_c.SizeBytes(), 0);
//      glColor3f(0,1,0);
//      pangolin::RenderVbo(vbo);

      if (showPcCurrent) {
        vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(), GL_FLOAT,
            3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
        cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
        pangolin::RenderVboCbo(vbo, cbo, true);
      }

      pangolin::glUnsetFrameOfReference();

      pangolin::glDrawAxis(0.3f);
//      vbo.Reinitialise(pangolin::GlArrayBuffer, pc_m.Area(), GL_FLOAT,
//          3, GL_DYNAMIC_DRAW);
//      vbo.Upload(pc_m.ptr_, pc_m.SizeBytes(), 0);
//      glColor3f(1,0,0);
//      pangolin::RenderVbo(vbo);

      if (showPcModel) {
        vbo.Reinitialise(pangolin::GlArrayBuffer, pcFull_m.Area(), GL_FLOAT,
            3, GL_DYNAMIC_DRAW);
        vbo.Upload(pcFull_m.ptr_, pcFull_m.SizeBytes(), 0);
        cbo.Upload(rgb_m.ptr_, rgb_m.SizeBytes(), 0);
        pangolin::RenderVboCbo(vbo, cbo, true);
      }

      glColor4f(1,0,0,1.);
      for (const auto& ass : assoc) {
        tdp::Vector3fda pc_c_in_m = T_wc*pc_c.GetCircular(ass.second);
//        tdp::glDrawLine(pc_m[ass.second], pc_c_in_m);
        tdp::glDrawLine(pl_w.GetCircular(ass.first).p_, pc_c_in_m);
      }
    }

    TOCK("Draw 3D");
    if (gui.verbose) std::cout << "draw 2D" << std::endl;
    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (containerTracking.IsShown()) {
      if (viewModel.IsShown()) {
        viewModel.SetImage(rgb_m);
      }
      if (viewCurrent.IsShown()) {
        viewCurrent.SetImage(rgb);
      }
    }
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

