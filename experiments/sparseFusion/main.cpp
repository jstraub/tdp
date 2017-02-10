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
#include <tdp/preproc/blur.h>
#include <tdp/gl/render.h>
#include <tdp/preproc/convert.h>
#include <tdp/preproc/plane.h>
#include <tdp/utils/timer.hpp>
#include <tdp/camera/projective_labels.h>
#include <tdp/ransac/ransac.h>
#include <tdp/utils/file.h>

#include <tdp/sampling/sample.hpp>
#include <tdp/sampling/vmf.hpp>
#include <tdp/sampling/vmfPrior.hpp>
#include <tdp/sampling/normal.hpp>

#include "planeHelpers.h"
#include "icpHelper.h"
#include "visHelper.h"

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

#define MAP_SIZE 100000

namespace tdp {

void AddToSortedIndexList(tdp::Vector5ida& ids, tdp::Vector5fda&
    values, int32_t id, float value) {
  for(int i=4; i>=0; --i) {
    if (value > values[i]) {
      if (i == 3) { 
        values[4] = value; 
        ids[4] = id;
      } else if (i == 2) {
        values[4] = values[3];
        values[3] = value; 
        ids[4] = ids[3];
        ids[3] = id; 
      } else if (i == 1) {
        values[4] = values[3];
        values[3] = values[2];
        values[2] = value; 
        ids[4] = ids[3];
        ids[3] = ids[2];
        ids[2] = id; 
      } else if (i == 0) {
        values[4] = values[3];
        values[3] = values[2];
        values[2] = values[1];
        values[1] = value; 
        ids[4] = ids[3];
        ids[3] = ids[2];
        ids[2] = ids[1];
        ids[1] = id; 
      }
      return;
    }
  }
  values[4] = values[3];
  values[3] = values[2];
  values[2] = values[1];
  values[1] = values[0];
  values[0] = value; 
  ids[4] = ids[3];
  ids[3] = ids[2];
  ids[2] = ids[1];
  ids[1] = ids[0];
  ids[0] = id; 
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

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewGrey(wc, hc);
  containerTracking.AddDisplay(viewGrey);
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
  pangolin::DataLog logEig;
  pangolin::Plotter plotEig(&logEig, -100.f,1.f, -5.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotEig);
  pangolin::DataLog logEv;
  pangolin::Plotter plotEv(&logEv, -100.f,1.f, -1.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotEv);
  gui.container().AddDisplay(plotters);

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

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> runLoopClosureGeom("ui.run loop closure geom",false,true);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updatePlanes("ui.update planes",true,true);
  pangolin::Var<bool> updateMap("ui.update map",false,true);
  pangolin::Var<bool> sampleMap("ui.sample map",true,true);
  pangolin::Var<bool> useMRF("ui.use MRF ",true,true);
  pangolin::Var<bool> warmStartICP("ui.warmstart ICP",false,true);
  pangolin::Var<bool> useDecomposedICP("ui.decomposed ICP",false,true);
  pangolin::Var<bool> useTexture("ui.use Tex in ICP",false,true);
  pangolin::Var<bool> useNormals("ui.use Ns in ICP",true,true);
  pangolin::Var<bool> useProj("ui.use proj in ICP",true,true);
  pangolin::Var<bool> pruneAssocByRender("ui.prune assoc by render",true,true);
  pangolin::Var<float> lambdaNs("ui.lamb Ns",0.1,0.0,1.);
  pangolin::Var<float> lambdaTex("ui.lamb Tex",0.1,0.0,1.);
  pangolin::Var<float> lambdaReg("ui.lamb Map Reg",.00,0.01,1.);
  pangolin::Var<float> alphaGrad("ui.alpha Grad",.01,0.0,1.);

  pangolin::Var<bool> icpReset("ui.reset icp",true,false);
  pangolin::Var<float> angleUniformityThr("ui.angle unif thr",5, 0, 90);
  pangolin::Var<float> angleThr("ui.angle Thr",15, -1, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.01,0,0.3);
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
  pangolin::Var<bool> showNormals("ui.show ns",false,true);
  pangolin::Var<bool> showAge("ui.show age",false,true);
  pangolin::Var<bool> showObs("ui.show # obs",false,true);
  pangolin::Var<bool> showCurv("ui.show curvature",false,true);
  pangolin::Var<bool> showSamples("ui.show Samples",true,true);
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
  pangolin::GlBuffer cbo_w(pangolin::GlArrayBuffer,MAP_SIZE,GL_UNSIGNED_BYTE,3);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<float> rs(MAP_SIZE); // radius of surfels
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_w(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector5ida> nn(MAP_SIZE);

  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  rs.Fill(NAN);
  rgb_w.Fill(tdp::Vector3bda::Zero());
  nn.Fill(tdp::Vector5ida::Ones()*-1);

  std::vector<std::pair<size_t, size_t>> mapNN;
  mapNN.reserve(MAP_SIZE*5);

  int32_t iReadCurW = 0;
  size_t frame = 0;

  tdp::ProjectiveAssociation<CameraT::NumParams, CameraT> projAssoc(cam, w, h);

  std::vector<std::pair<size_t, size_t>> assoc;
  assoc.reserve(10000);

  uint32_t numObs = 0;

  float lambDPvMFmeans = cos(55.*M_PI/180.);
  tdp::DPvMFmeansSimple3fda dpvmf(lambDPvMFmeans);

  std::vector<std::vector<uint32_t>> invInd;
  std::vector<size_t> id_w;
  id_w.reserve(MAP_SIZE);

//  std::random_device rd;
  std::mt19937 gen(19023);

  mask.Fill(0);

  tdp::ThreadedValue<bool> runTopologyThread(true);
  tdp::ThreadedValue<bool> runSampling(true);

  std::mutex pl_wLock;
  std::mutex nnLock;
  std::mutex mapLock;
  std::mutex dpvmfLock;
  std::thread topology([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
    int32_t sizeToRead = 0;
    tdp::Vector5fda values;
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
        tdp::Vector5ida& ids = nn[iReadNext];
        ids = tdp::Vector5ida::Ones()*(-1);
        for (size_t i=0; i<sizeToRead; ++i) {
          if (i != iReadNext) {
            float dist = (pl.p_-pl_w.GetCircular(i).p_).squaredNorm();
            tdp::AddToSortedIndexList(ids, values, i, dist);
          }
        }
        // just for visualization
        if (mapNN.size() < 5*iReadNext) {
          for (int i=0; i<5; ++i) 
            mapNN.emplace_back(iReadNext, ids[i]);
        } else {
          for (int i=0; i<5; ++i) 
            mapNN[iReadNext*5+i] = std::pair<size_t,size_t>(iReadNext, ids[i]);
        }
        iReadNext = (iReadNext+1)%sizeToRead;
        {
          std::lock_guard<std::mutex> lock(nnLock); 
          nn.iInsert_ = iReadNext;
        }
      }
    };
  });


  std::mutex vmfsLock;
  std::mt19937 rnd(910481);
  float logAlpha = log(10.);
  float lambdaMRF = 0.1;
  float tauO = 30.;
  Eigen::Matrix3f SigmaO = 0.0001*Eigen::Matrix3f::Identity();
  Eigen::Matrix3f InfoO = 10000.*Eigen::Matrix3f::Identity();
  vMFprior<float> base(Eigen::Vector3f(0,0,1), 1., 0.5);
  std::vector<vMF<float,3>> vmfs;
  vmfs.push_back(base.sample(rnd));

  tdp::ManagedHostCircularBuffer<uint32_t> zS(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> nS(MAP_SIZE);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pS(MAP_SIZE);
  nS.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  pS.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  zS.Fill(999); //std::numeric_limits<uint32_t>::max());
  tdp::ManagedHostCircularBuffer<tdp::Vector4fda> vmfSS(1000);
  vmfSS.Fill(tdp::Vector4fda::Zero());

  std::thread sampling([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
//    std::random_device rd_;
    std::mt19937 rnd(0);
    while(runSampling.Get()) {
    if (sampleMap) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        iRead = nn.iRead_;
        iInsert = nn.iInsert_;
      }
      pS.iInsert_ = nn.iInsert_;
      nS.iInsert_ = nn.iInsert_;
      // sample normals using dpvmf and observations from planes
      size_t K = vmfs.size();
      vmfSS.Fill(tdp::Vector4fda::Zero());
      for (int32_t iReadNext = 0; iReadNext!=iInsert;
        iReadNext=(iReadNext+1)%nn.w_) {
        tdp::Vector3fda& ni = nS.GetCircular(iReadNext);
        uint32_t& zi = zS.GetCircular(iReadNext);
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        Eigen::Vector3f mu = pl.n_*tauO;
        if (zi < K) {
          mu += vmfs[zi].mu_*vmfs[zi].tau_;
        }
        ni = vMF<float,3>(mu).sample(rnd);
        vmfSS[zi].topRows<3>() += ni;
        vmfSS[zi](3) ++;
      }
      // sample dpvmf labels
      for (int32_t iReadNext = 0; iReadNext!=iInsert;
        iReadNext=(iReadNext+1)%nn.w_) {

        Eigen::VectorXf logPdfs(K+1);
        Eigen::VectorXf pdfs(K+1);

        tdp::Vector3fda& ni = nS.GetCircular(iReadNext);
        uint32_t& zi = zS.GetCircular(iReadNext);
        tdp::Vector5ida& ids = nn.GetCircular(iReadNext);

        Eigen::VectorXf neighNs = Eigen::VectorXf::Zero(K);
        for (int i=0; i<5; ++i) {
          if (ids[i] > -1  && zS[ids[i]] < K) {
            neighNs[zS[ids[i]]] += 1.f;
          }
        }
        for (size_t k=0; k<K; ++k) {
          if (useMRF) 
            logPdfs[k] = lambdaMRF*(neighNs[k]-5.);
          else 
            logPdfs[k] = 0.;
          if (zi == k) {
            logPdfs[k] += log(vmfSS[k](3)-1)+vmfs[k].logPdf(ni);
          } else {
            logPdfs[k] += log(vmfSS[k](3))+vmfs[k].logPdf(ni);
          }
        }
        logPdfs[K] = logAlpha + base.logMarginal(ni);
        logPdfs = logPdfs.array() - logSumExp<float>(logPdfs);
        pdfs = logPdfs.array().exp();
        size_t zPrev = zi;
        zi = sampleDisc(pdfs, rnd);
        //      std::cout << z[i] << " " << K << ": " << pdfs.transpose() << std::endl;
        if (zi == K) {
          vmfsLock.lock();
          vmfs.push_back(base.posterior(ni,1).sample(rnd));
          vmfsLock.unlock();
          K++;
        }
        if (zPrev != zi) {
          vmfSS[zPrev].topRows<3>() -= ni;
          vmfSS[zPrev](3) --;
          vmfSS[zi].topRows<3>() += ni;
          vmfSS[zi](3) ++;
        }
      }
      // sample dpvmf parameters
      {
        std::lock_guard<std::mutex> lock(vmfsLock);
        for (size_t k=0; k<K; ++k) {
          if (vmfSS[k](3) > 0) {
            vmfs[k] = base.posterior(vmfSS[k]).sample(rnd);
          }
        }
      }
      std::cout << "counts " << K << ": ";
      for (size_t k=0; k<K; ++k) 
        if (vmfSS[k](3) > 0) 
          std::cout << vmfSS[k](3) << " ";
      std::cout << "\ttaus: " ;
      for (size_t k=0; k<K; ++k) 
        if (vmfSS[k](3) > 0) 
          std::cout << vmfs[k].tau_ << " ";
      std::cout << std::endl;
      // sample points
      for (int32_t iReadNext = 0; iReadNext!=iInsert;
        iReadNext=(iReadNext+1)%nn.w_) {
        tdp::Vector3fda& pi = pS.GetCircular(iReadNext);
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        tdp::Vector5ida& ids = nn.GetCircular(iReadNext);

        Eigen::Matrix3f SigmaPl;
        Eigen::Matrix3f Info =  InfoO;
//        Eigen::Vector3f xi = SigmaO.ldlt().solve(pl.p_);
        Eigen::Vector3f xi = InfoO*pl.p_;
        for (int i=0; i<5; ++i) {
          if (ids[i] > -1  && zS[ids[i]] < K) {
            SigmaPl = vmfs[zS[ids[i]]].mu_*vmfs[zS[ids[i]]].mu_.transpose();
            Info += SigmaPl;
            xi += SigmaPl*pS[ids[i]];
          }
        }
        Eigen::Matrix3f Sigma = Info.inverse();
        Eigen::Vector3f mu = Sigma*xi;
//        std::cout << xi.transpose() << " " << mu.transpose() << std::endl;
        pi = Normal<float,3>(mu, Sigma).sample(rnd);
      }

    }
    };
  });


  std::vector<uint32_t> idsCur;
  idsCur.reserve(w*h);

  tdp::ConfigICP cfgIcp;
  size_t numNonPlanar = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    cfgIcp.distThr = distThr;
    cfgIcp.p2plThr = p2plThr; 
    cfgIcp.dotThr = cos(angleThr*M_PI/180.);
    cfgIcp.condEntropyThr = condEntropyThr;
    cfgIcp.negLogEvThr = negLogEvThr;
    cfgIcp.HThr = HThr;
    cfgIcp.lambdaNs = lambdaNs;
    cfgIcp.lambdaTex = lambdaTex;
    cfgIcp.useTexture = useTexture;
    cfgIcp.useNormals = useNormals;
    cfgIcp.numRotThr = numRotThr;

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

//        tdp::UniformResampleEmptyPartsOfMask(mask, W, subsample, gen, 16, 16);
//        tdp::UniformResampleMask(pc, cam, mask, W, subsample, gen, 16, 16);
      tdp::UniformResampleEmptyPartsOfMask(pc, cam, mask, W,
          subsample, gen, 32, 32, w, h);
      TOCK("mask");
      {
        iReadCurW = pl_w.iInsert_;
        std::lock_guard<std::mutex> lock(pl_wLock); 
        TICK("normals");
//        tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, W, pts,
//            orientation);
        ExtractPlanes(pc, rgb, grey, greyFl, gradGrey,
            mask, W, frame, T_wc, cam, dpc, pl_w, pc_w, rgb_w,
            n_w, rs);
        TOCK("normals");

        TICK("add to model");
        for (int32_t i = iReadCurW; i != pl_w.iInsert_; i = (i+1)%pl_w.w_) {
          tdp::Plane& pl = pl_w[i];
          if (pl.curvature_ > curvThr) {
            pl.z_ = 0xFFFF; // mark high curvature cluster as outlier
            numNonPlanar ++;
          }
          dpvmf.addObservation(&pl.n_, &pl.z_);
          int32_t kMax = -1;
          uint32_t nMax = 0;
          for (size_t k=0; k<dpvmf.GetK(); ++k) {
            if (k==pl.z_) continue;
            if (nMax < dpvmf.GetNs()[k]) {
              nMax = dpvmf.GetNs()[k];
              kMax = k;
            }
          }
          if (kMax >= 0) {
            pl.dir_ = dpvmf.GetCenter(kMax);
          }
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
        << " " << dpvmf.GetZs().size() << " non planar: " 
        << numNonPlanar << std::endl;
      TICK("dpvmf");
      dpvmfLock.lock();
      dpvmf.iterateToConvergence(100, 1e-6);
      dpvmfLock.unlock();
      for (size_t k=0; k<dpvmf.GetK()+1; ++k) {
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
          uint32_t k = std::min((uint32_t)(*dpvmf.GetZs()[i]), dpvmf.GetK());
          if (invInd[k].size() < 10000)
            invInd[k].push_back(i);
        }
      } else {      
        // use all ids in the current map
        for (auto i : id_w) {
          uint32_t k = std::min((uint32_t)(*dpvmf.GetZs()[i]), dpvmf.GetK());
          if (invInd[k].size() < 10000)
            invInd[k].push_back(i);
        }
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
    cuGreyFlSmooth.CopyFrom(cuGreyFl);
//    tdp::Blur5(cuGreyFl,cuGreyFlSmooth, 10.);
    tdp::Convert(cuGreyFlSmooth, cuGrey, 255.);
    grey.CopyFrom(cuGrey);
    greyFl.CopyFrom(cuGreyFlSmooth);
    tdp::Gradient(cuGreyFlSmooth, cuGreyDu, cuGreyDv, cuGradGrey);

    n.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    size_t numProjected =0;
    trackingGood = false;
    if (frame > 1 && runTracking && !gui.finished()) { // tracking
      TICK("icp");
      Eigen::Matrix<float,6,6> A;
      Eigen::Matrix<float,6,1> b;
      Eigen::Matrix<float,6,1> Ai;
      Eigen::Matrix<float,6,1> x;
      Eigen::Matrix<float,3,3> At;
      Eigen::Matrix<float,3,1> xt;
      uint32_t numInl = 0;

      std::uniform_int_distribution<> dis(0, dpvmf.GetK());
      
      tdp::SE3f T_wcPrev = T_wc.rotation();
      std::vector<size_t> indK(dpvmf.GetK()+1,0);
      for (size_t it = 0; it < maxIt; ++it) {
        mask.Fill(0);
        assoc.clear();
//        pc_c.MarkRead();
//        n_c.MarkRead();
        indK = std::vector<size_t>(dpvmf.GetK()+1,0);
        numProjected = 0;

        float err = 0.;
        float H = 1e10;
        if (useDecomposedICP) {
          tdp::SO3f R_wcBefore = T_wc.rotation();
          tdp::SO3f R_wc;
          // TODO need better termination criterion
          IncrementalOpRot(pc, dpc, n, curv, invInd, T_wcPrev, T_wc, cam, cfgIcp, W,
              indK, frame, mask, pl_w, assoc, numProjected, R_wc);
          T_wc.rotation() = R_wc;
          Eigen::Matrix<float,3,1> xR = R_wcBefore.Log(R_wc);
          numInl = 0;
          IncrementalICPTranslation( pc, dpc, n,  grey, curv, invInd, cam,
              cfgIcp, W, indK, frame, mask, pl_w, assoc, numProjected,
              numInl, T_wc, H, At, xt, err);
          if (gui.verbose) {
            std::cout << "\tit " << it << ": err=" << err 
              << "\t# inliers: " << numInl
              << "\t|x|: " << xR.norm()*180./M_PI 
              << " " << xt.norm()
              << std::endl;
          }
          // TODO needs to take into account rotation as well
          if (xR.norm()*180./M_PI < icpdRThr
              && xt.norm() < icpdtThr
              && tdp::CheckEntropyTermination(At, H, cfgIcp.HThr, 0.f,
                cfgIcp.negLogEvThr, H)) {
            std::cout << numInl << " " << numObs << " " << numProjected << std::endl;
            break;
          }
        } else {
          if (warmStartICP) {
            tdp::SO3f R_wc;
            // TODO need better termination criterion
            IncrementalOpRot(pc, dpc, n, curv, invInd, T_wcPrev, T_wc,
                cam, cfgIcp, W, indK, frame, mask, pl_w, assoc,
                numProjected, R_wc);
            T_wc.rotation() = R_wc;
          }
          numInl = 0;
          IncrementalFullICP( pc, dpc, n,  grey, curv, invInd, cam,
              cfgIcp, W, indK, frame, mask, pl_w, assoc, numProjected,
              numInl, T_wc, H, A, x, err);

          if (gui.verbose) {
            std::cout << "\tit " << it << ": err=" << err 
              << "\t# inliers: " << numInl
              << "\t|x|: " << x.topRows(3).norm()*180./M_PI 
              << " " <<  x.bottomRows(3).norm()
              << std::endl;
          }
          if (x.topRows<3>().norm()*180./M_PI < icpdRThr
              && x.bottomRows<3>().norm() < icpdtThr
              && tdp::CheckEntropyTermination(A, H, cfgIcp.HThr, 0.f,
                cfgIcp.negLogEvThr, H)) {
            std::cout << numInl << " " << numObs << " " << numProjected << std::endl;
            break;
          }
        }
        numObs = assoc.size();
      }
      for (size_t k=0; k<indK.size(); ++k) {
        std::cout << "used different directions " << k << "/" 
          << (dpvmf.GetK()+1) << ": " << indK[k] 
          << " of " << invInd[k].size() << std::endl;
      }
      float H;
      if (useDecomposedICP) {
        Eigen::Matrix<float,3,1> ev = At.eigenvalues().real();
        H = -ev.array().log().sum();
        std::cout << " H " << H << " neg log evs " << 
          ev.array().log().matrix().transpose() << std::endl;
      } else {
        Eigen::Matrix<float,6,1> ev = A.eigenvalues().real();
        H = -ev.array().log().sum();
        std::cout << " H " << H << " neg log evs " << 
          -ev.array().log().matrix().transpose() << std::endl;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
        Eigen::Matrix<float,6,6> Q = eig.eigenvectors();
        //      for (size_t k=0; k<dpvmf.GetK(); ++k) {
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
      } 
      T_wcs.push_back(T_wc);
      trackingGood = H <= HThr && numInl > 10;
      logObs.Log(log(numObs)/log(10.), log(numInl)/log(10.), 
          log(numProjected)/log(10.), log(pl_w.SizeToRead())/log(10));
      TOCK("icp");
      if (trackingGood) {
        std::cout << "tracking good" << std::endl;
      }

      if (updatePlanes && trackingGood) {
        std::lock_guard<std::mutex> mapGuard(mapLock);
        TICK("update planes");
        for (const auto& ass : assoc) {
          int32_t u = ass.second%w;
          int32_t v = ass.second/w;
          tdp::Vector3fda pc_c_in_w = T_wc*pc(u,v);
          tdp::Vector3fda n_c_in_w = T_wc.rotation()*n(u,v);
          pl_w.GetCircular(ass.first).AddObs(pc_c_in_w, n_c_in_w);
          n_w.GetCircular(ass.first) = pl_w.GetCircular(ass.first).n_;
          pc_w.GetCircular(ass.first) = pl_w.GetCircular(ass.first).p_;
        }
        TOCK("update planes");
      }
    }

    if (runLoopClosureGeom && dpvmf.GetK()>2) {
      tdp::ManagedDPvMFmeansSimple3fda dpvmfCur(lambDPvMFmeans);
      for (const auto& ass : assoc) {
        dpvmfCur.addObservation(n(ass.second%w,ass.second/w));
      }
      dpvmfCur.iterateToConvergence(100, 1e-6);
      if (dpvmfCur.GetK() > 2) {
        std::vector<size_t> idsW(dpvmf.GetK());
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
            N +=  dpvmfCur.GetCenter(idsC[i]) * dpvmf.GetCenter(idsW[i]).transpose();
          }
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
      pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wc.matrix(), 0.1f);

      if (showLoopClose) {
        glColor4f(1.,0.,0.,1.0);
        pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wcRansac.matrix(), 0.1f);
      }
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,20, 0.03f);

      std::cout << "uploading pc" << std::endl;
      if (showSamples) {
        vbo_w.Upload(pS.ptr_, pS.SizeToRead(), 0);
        nbo_w.Upload(nS.ptr_, nS.SizeToRead(), 0);
      } else {
        vbo_w.Upload(pc_w.ptr_, pc_w.SizeToRead(), 0);
        nbo_w.Upload(n_w.ptr_, n_w.SizeBytes(), 0);
      }
      cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeBytes(), 0);
      std::cout << "uploading pc done" << std::endl;

      if (showFullPc) {
        pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
        pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
        // TODO I should not need to upload all of pc_w everytime;
        // might break things though
        // I do ned to upload points because they get updated; I
        // wouldnt have to with the color
        if ((!showAge && !showObs && !showSurfels && !showCurv) 
            || pl_w.SizeToRead() == 0) {
          pangolin::RenderVboCbo(vbo_w, cbo_w, true);
        } else if (showAge || showObs || showCurv) {
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
          valuebo.Reinitialise(pangolin::GlArrayBuffer, age.Area(),
              GL_FLOAT, 1, GL_DYNAMIC_DRAW);
          valuebo.Upload(age.ptr_,  age.SizeBytes(), 0);

          std::pair<float,float> minMaxAge = age.MinMax();
          tdp::RenderVboValuebo(vbo_w, valuebo, minMaxAge.first,
              minMaxAge.second, P, MV);
        } else if (showSurfels) {
          std::cout << "rbo upload " << std::endl;
          rbo.Upload(rs.ptr_, rs.SizeBytes(), 0);
          std::cout << "render surfels" << std::endl;
          tdp::RenderSurfels(vbo_w, nbo_w, cbo_w, rbo, dMax, P, MV);
        }
        if (showNN) {
          glColor4f(0.3,0.3,0.3,0.3);
          for (auto& ass : mapNN) {
            if (ass.second >= 0)
              tdp::glDrawLine(pl_w[ass.first].p_, pl_w[ass.second].p_);
          }
        }
        if (showLoopClose) {
        }
      }

      if (showNormals) {
        std::cout << "render normals local" << std::endl;
        tdp::ShowCurrentNormals(pc, n, assoc, T_wc, scale);
        std::cout << "render normals global " << n_w.SizeToRead() << std::endl;
        tdp::ShowGlobalNormals(pc_w, n_w, scale, step);
        std::cout << "render normals done" << std::endl;
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
//        tdp::Vector3fda pc_c_in_m = T_wc*pc_c.GetCircular(ass.second);
        tdp::Vector3fda pc_c_in_m = T_wc*pc(ass.second%w,ass.second/w);
        tdp::glDrawLine(pl_w.GetCircular(ass.first).p_, pc_c_in_m);
      }
    }

    if (viewNormals.IsShown()) {
      Eigen::Matrix4f Tview = s_cam.GetModelViewMatrix();
      Tview(0,3) = 0.; Tview(1,3) = 0.; Tview(2,3) = -2.2;
      normalsCam.GetModelViewMatrix() = Tview;
      viewNormals.Activate(normalsCam);
      glColor4f(0,0,1,0.5);
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
    if (viewCurrent.IsShown()) {
//      if (showFAST && !useFAST) {
//        tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, W, pts,
//            orientation);
//      }
      viewCurrent.SetImage(rgb);
      glColor3f(1,0,0);
      for (size_t u=0; u<rgb.w_; ++u)
        for (size_t v=0; v<rgb.h_; ++v) {
          if (mask(u,v)) {
            pangolin::glDrawCircle(u,v,1);
          }
        }
    }

    if (containerTracking.IsShown()) {
      if (viewMask.IsShown()) {
        viewMask.SetImage(mask);
      }
      if (viewGrey.IsShown()) {
        viewGrey.SetImage(grey);
      }
    }
    if (!gui.finished()) {
      std::cout << "scroll plots" << std::endl;
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
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }

//  imuInterp.Stop();
//  if (imu) imu->Stop();
//  delete imu;
//  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

