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
#include <tdp/preproc/convert.h>
#include <tdp/preproc/plane.h>
#include <tdp/features/lsh.h>
#include <tdp/utils/timer.hpp>
#include <tdp/camera/projective_labels.h>
#include <tdp/ransac/ransac.h>
#include <tdp/utils/file.h>

#include "planeHelpers.h"
#include "featureHelper.h"
#include "icpHelper.h"
#include "visHelper.h"

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

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
  pangolin::Var<bool> useFAST("ui.use FAST",false,true);

  pangolin::Var<bool> runTracking("ui.run tracking",true,true);
  pangolin::Var<bool> runLoopClosure("ui.run loop closure",false,true);
  pangolin::Var<bool> runLoopClosureGeom("ui.run loop closure geom",false,true);
  pangolin::Var<bool> trackingGood("ui.tracking good",false,true);
  pangolin::Var<bool> runMapping("ui.run mapping",true,true);
  pangolin::Var<bool> updatePlanes("ui.update planes",true,true);
  pangolin::Var<bool> updateMap("ui.update map",false,true);
  pangolin::Var<bool> warmStartICP("ui.warmstart ICP",false,true);
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
//  pangolin::Var<float> angleThr("ui.angle Thr",-1, -1, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.01,0,0.3);
  pangolin::Var<float> distThr("ui.dist Thr",0.1,0,0.3);
  pangolin::Var<float> curvThr("ui.curv Thr",0.06,0.01,1.0);
  pangolin::Var<float> assocDistThr("ui.assoc dist Thr",0.1,0,0.3);
  pangolin::Var<float> HThr("ui.H Thr",-12.,-20.,-8.);
  pangolin::Var<float> negLogEvThr("ui.neg log ev Thr",-0.,-2.,1.);
  pangolin::Var<float> condEntropyThr("ui.rel log dH ", 1.e-3,1.e-3,1e-2);
  pangolin::Var<float> icpdRThr("ui.dR Thr",0.01,0.1,0.1);
  pangolin::Var<float> icpdtThr("ui.dt Thr",0.001,0.01,0.001);
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
  pangolin::Var<bool> showSurfels("ui.show surfels",true,true);
  pangolin::Var<bool> showNN("ui.show NN",true,true);
  pangolin::Var<bool> showLoopClose("ui.show loopClose",false,true);
  pangolin::Var<int> step("ui.step",10,0,100);

  pangolin::Var<bool> showFAST("ui.show FAST",true,true);
  pangolin::Var<int> fastB("ui.FAST b",30,0,100);
  pangolin::Var<float> harrisThr("ui.harris thr",0.1,0.001,2.0);
  pangolin::Var<float> kappaHarris("ui.kappa harris",0.08,0.04,0.15);
  pangolin::Var<int> briefMatchThr("ui.BRIEF match",65,0,100);
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

  pangolin::GlBuffer vbo_w(pangolin::GlArrayBuffer,1000000,GL_FLOAT,3);
  pangolin::GlBuffer nbo_w(pangolin::GlArrayBuffer,1000000,GL_FLOAT,3);
  pangolin::GlBuffer rbo(pangolin::GlArrayBuffer,1000000,GL_FLOAT,1);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(1000000);
  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));
  tdp::ManagedHostCircularBuffer<float> rs(1000000); // radius of surfels
  rs.Fill(NAN);
  pangolin::GlBuffer cbo_w(pangolin::GlArrayBuffer,1000000,GL_UNSIGNED_BYTE,3);
  tdp::ManagedHostCircularBuffer<tdp::Vector3bda> rgb_w(1000000);
  rgb_w.Fill(tdp::Vector3bda::Zero());

  tdp::ManagedHostCircularBuffer<tdp::Plane> pl_w(1000000);
//  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_c(1000000);
//  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_c(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> n_w(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector5ida> nn(1000000);
  nn.Fill(tdp::Vector5ida::Ones()*-1);
  tdp::ManagedHostCircularBuffer<tdp::Vector5fda> mapObsDot(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector5fda> mapObsP2Pl(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc0_w(1000000);

  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> Jn_w(1000000);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> Jp_w(1000000);

  std::vector<std::pair<size_t, size_t>> mapNN;
  mapNN.reserve(10000000);

  int32_t iReadCurW = 0;
  size_t frame = 0;

  tdp::ManagedLshForest<14> lsh(11);
  tdp::ManagedHostImage<tdp::Brief> descs;
  tdp::ManagedHostImage<tdp::Vector2ida> pts;
  tdp::ManagedHostImage<float> orientation;

  tdp::ProjectiveAssociation<CameraT::NumParams, CameraT> projAssoc(cam, w, h);

  std::vector<std::pair<size_t, size_t>> assoc;
  assoc.reserve(10000);

  uint32_t numObs = 0;

  float lambDPvMFmeans = cos(55.*M_PI/180.);
  tdp::DPvMFmeansSimple3fda dpvmf(lambDPvMFmeans);

  std::vector<std::vector<uint32_t>> invInd;
  std::vector<size_t> id_w;
  id_w.reserve(1000000);

//  std::random_device rd;
  std::mt19937 gen(19023);

  mask.Fill(0);

  std::mutex pl_wLock;
  std::mutex nnLock;
  std::mutex mapLock;
  std::mutex dpvmfLock;
  std::thread mapping([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
    int32_t sizeToRead = 0;
    tdp::Vector5fda values;
    while(42) {
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
        // for map constraints
        // TODO: should be updated as pairs are reobserved
        for (int i=0; i<5; ++i) {
          mapObsDot[iReadNext][i] = pl.n_.dot(pl_w[ids[i]].n_);
          mapObsP2Pl[iReadNext][i] = pl.p2plDist(pl_w[ids[i]].p_);
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
          mapObsDot.iInsert_ = iReadNext;
          mapObsP2Pl.iInsert_ = iReadNext;
          nn.iInsert_ = iReadNext;
        }
      }
    };
  });

  std::thread regularization([&]() {
    int32_t iRead = 0;
    int32_t iInsert = 0;
    int32_t iReadNext = 0;
//    std::random_device rd_;
    std::mt19937 gen_(0);
    while(42) {
    if (updateMap) {
      {
        std::lock_guard<std::mutex> lock(nnLock); 
        iRead = nn.iRead_;
        iInsert = nn.iInsert_;
      }
      // compute gradient
      for (int32_t iReadNext = 0; iReadNext!=iInsert;
        iReadNext=(iReadNext+1)%nn.w_) {
        tdp::Vector5ida& ids = nn.GetCircular(iReadNext);
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        tdp::Vector3fda& Jn = Jn_w[iReadNext];
        tdp::Vector3fda& Jp = Jp_w[iReadNext];
        Jn = tdp::Vector3fda::Zero();
        Jp = tdp::Vector3fda::Zero();
        for (int i=0; i<5; ++i) {
          if (ids[i] > -1) {
            const tdp::Plane& plO = pl_w[ids[i]];
            Jn += 2.*(pl.n_.dot(plO.n_)-mapObsDot[iReadNext][i])*plO.n_;
            Jn += 2.*(pl.p2plDist(plO.p_)-mapObsP2Pl[iReadNext][i])*(plO.p_-pl.p_);
            if (pl.curvature_ < curvThr) {
              dpvmfLock.lock();
              Jn += -lambdaReg*dpvmf.GetCenter(pl.z_);
              dpvmfLock.unlock();
            }
            Jp += -2.*(pl.p2plDist(plO.p_)-mapObsP2Pl[iReadNext][i])*pl.n_;
            Jp += -2.*(pc0_w[iReadNext] - pl.p_);
          }
        }
      }
      // apply gradient
      for (int32_t iReadNext = 0; iReadNext!=iInsert;
        iReadNext=(iReadNext+1)%nn.w_) {
        tdp::Plane& pl = pl_w.GetCircular(iReadNext);
        tdp::Vector3fda& Jn = Jn_w[iReadNext];
        tdp::Vector3fda& Jp = Jp_w[iReadNext];
        std::lock_guard<std::mutex> mapGuard(mapLock);
        pl.n_ = (pl.n_- alphaGrad * Jn).normalized();
        pl.p_ -= alphaGrad * Jp;
        pc_w[iReadNext] = pl.p_;
        n_w[iReadNext] = pl.n_;
      }
//      std::cout << "map updated " << iReadNext << " " 
//        << (alphaGrad * Jn.transpose()) << "; "
//        << (alphaGrad * Jp.transpose()) << std::endl;
    }
    };
  });


  std::vector<uint32_t> idsCur;
  idsCur.reserve(w*h);
  std::vector<int32_t> assocBA;
  std::vector<tdp::Brief> featsB;
  std::vector<tdp::Brief> featsA;
  assocBA.reserve(4*subsample*w*h);
  featsA.reserve( 4*subsample*w*h);
  featsB.reserve( 4*subsample*w*h);

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

    if (runLoopClosure.GuiChanged()) {
      showLoopClose = runLoopClosure;
    }
    if (runLoopClosureGeom.GuiChanged()) {
      showLoopClose = runLoopClosureGeom;
    }
    if (pangolin::Pushed(icpReset)) {
      T_wc = tdp::SE3f();
    }
    if (runLoopClosure) {
      // TODO: I did not set orientation propperly when I tried BRIEF
      // simply on sampled plane locations!!! Try that again.
      TICK("match briefs");
      assocBA.clear(); featsA.clear(); featsB.clear();
      tdp::Brief* featB;
      tdp::Brief featA;
      int dist;
      std::cout << "have " << pts.Area() << " key points" << std::endl;
      for (size_t i=0; i<pts.Area(); ++i) {
        featA.p_c_ = pc(pts[i](0),pts[i](1));
        if (!tdp::IsValidData(featA.p_c_)) 
          continue;
        featA.pt_ = pts[i];
        featA.orientation_ = orientation[i];
        if (tdp::ExtractBrief(grey, featA)
            && lsh.SearchBest(featA,dist,featB) 
            && dist < briefMatchThr) {
//          std::cout << i << " " << assocBA.size() << ": " << dist 
//            << " " << featA.p_c_.transpose()
//            << " " << featB->p_c_.transpose()
//            << std::endl;
          assocBA.push_back(assocBA.size());
          featsB.push_back(*featB);
          featsA.push_back(featA);
//          std::cout << " " << featsA.back().p_c_.transpose()
//            << " " << featsB.back().p_c_.transpose()
//            << std::endl;
        }
      }
      TOCK("match briefs");
      if (assocBA.size() >= ransacInlierThr) {
        TICK("RANSAC");
        tdp::P3PBrief p3p;
        tdp::Ransac<tdp::Brief> ransac(&p3p);
        size_t numInliers = 0;
        tdp::SE3f T_ab = ransac.Compute(featsA, featsB, assocBA, ransacMaxIt,
            ransacThr, numInliers);
        TOCK("RANSAC");

        std::cout << "matches: " << assocBA.size() 
          << " " << assocBA.size()/(float)pl_w.SizeToRead(iReadCurW)
          << "%;  after RANSAC "
          << numInliers << " " << numInliers/(float)assocBA.size()
          << std::endl;
        std::cout << T_wc.Log(T_ab.Inverse()).transpose() << std::endl;
        if (numInliers > ransacInlierThr) {
          T_wcRansac = T_ab.Inverse();
        } else {
          T_wcRansac.translation() << 999,999,999;
        }
      }
    }

    if (!gui.paused() && !gui.finished()
        && frame > 0
        && (runMapping || frame == 1) 
        && (trackingGood || frame < 10)) { // add new observations
      TICK("mask");

      // update plane features
      tdp::Brief feat;
      size_t numAdded = 0;
      for (size_t i=0; i<z.Area(); ++i) {
        if (z[i] > 0 && tdp::IsValidData(pc[i])
            && pl_w.GetCircular(z[i]-1).feat_.desc_.sum() > 0)  {
          if (ExtractClosestBrief(pc, grey, pts, orientation, 
                T_wc.Inverse()*pl_w.GetCircular(z[i]-1).p_,
                T_wc.rotation().Inverse()*pl_w.GetCircular(z[i]-1).n_,
                T_wc, cam, W, 
              i%mask.w_, i/mask.w_, feat)) {
            pl_w.GetCircular(z[i]-1).feat_ = feat;
            numAdded++;
          }
        }
      }
      std::cout << "num features added to planes" << std::endl;

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

//      tdp::RandomMaskCpu(mask, perc, W*dMax);
//      tdp::UniformResampleMask(mask, W, subsample, gen, 4, 4);
      if (useFAST) {
        tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, 18, pts,
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
            subsample, gen, 32, 32, w, h);
      }
      TOCK("mask");
      {
        iReadCurW = pl_w.iInsert_;
        std::lock_guard<std::mutex> lock(pl_wLock); 
        TICK("normals");
//        tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, W, pts,
//            orientation);
        ExtractPlanes(pc, rgb, grey, greyFl, gradGrey, pts,
            orientation, mask, W, frame, T_wc, cam, dpc, pl_w, pc_w, pc0_w, rgb_w,
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
          lsh.Insert(pl.feat_);
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

    if (runLoopClosure) {
      TICK("FAST");
      tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, W, pts,
          orientation);
      TOCK("FAST");
    }

    n.Fill(tdp::Vector3fda(NAN,NAN,NAN));
    TOCK("Setup");

    size_t numProjected =0;
    trackingGood = false;
    if (frame > 1 && runTracking && !gui.finished()) { // tracking
      TICK("icp");
      Eigen::Matrix<float,6,6> A;
      Eigen::Matrix<float,6,1> b;
      Eigen::Matrix<float,6,1> Ai;
      uint32_t numInl = 0;

      std::uniform_int_distribution<> dis(0, dpvmf.GetK());
      
      std::vector<size_t> indK(dpvmf.GetK()+1,0);
      for (size_t it = 0; it < maxIt; ++it) {
        mask.Fill(0);
        assoc.clear();
//        pc_c.MarkRead();
//        n_c.MarkRead();
        indK = std::vector<size_t>(dpvmf.GetK()+1,0);
        numProjected = 0;

        if (warmStartICP) {
          tdp::SO3f R_wc;
          IncrementalOpRot(pc, dpc, n, curv, invInd, T_wc, cam, cfgIcp, W,
              indK, frame, mask, pl_w, assoc, numProjected, R_wc);
          T_wc.rotation() = R_wc;
        }
        
        float err = 0.;
        float H = 1e10;
        numInl = 0;
        IncrementalFullICP( pc, dpc, n,  grey, curv, invInd, cam,
            cfgIcp, W, indK, frame, mask, pl_w, assoc, numProjected,
            numInl, T_wc, H, A, b, err);
        numObs = assoc.size();

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
            && tdp::CheckEntropyTermination(A, H, cfgIcp.HThr, 0.f,
              cfgIcp.negLogEvThr, H)) {
          std::cout << numInl << " " << numObs << " " << numProjected << std::endl;
          break;
        }
      }
      for (size_t k=0; k<indK.size(); ++k) {
        std::cout << "used different directions " << k << "/" 
          << (dpvmf.GetK()+1) << ": " << indK[k] 
          << " of " << invInd[k].size() << std::endl;
      }
      Sigma_mc = A.inverse();
      logObs.Log(log(numObs)/log(10.), log(numInl)/log(10.), 
          log(numProjected)/log(10.), log(pl_w.SizeToRead())/log(10));
      Eigen::Matrix<float,6,1> ev = Sigma_mc.eigenvalues().real();
      float H = ev.array().log().sum();
      std::cout << " H " << H << " neg log evs " << 
        ev.array().log().matrix().transpose() << std::endl;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float,6,6>> eig(A);
      Eigen::Matrix<float,6,6> Q = eig.eigenvectors();
//      for (size_t k=0; k<dpvmf.GetK(); ++k) {
//        Eigen::Matrix<float,6,1> Ai;
//        Ai << Eigen::Vector3f::Zero(), dpvmf.GetCenter(k);
//        std::cout << "k " << k << std::endl;
//        std::cout << (Q.transpose()*Ai*Ai.transpose()*Q).diagonal().transpose() << std::endl;
//      }

      logEntropy.Log(H);
      logEig.Log(ev.array().log().matrix());
      Eigen::Matrix<float,6,1> q0 = Q.col(0);
      uint32_t maxId = 0;
      q0.array().abs().maxCoeff(&maxId);
      q0 *= (q0(maxId) > 0? 1.: -1.);
      logEv.Log(q0);
      T_wcs.push_back(T_wc);
      trackingGood = H <= HThr && numInl > 10;
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
      pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wc.matrix(), 0.1f);

      if (showLoopClose) {
        glColor4f(1.,0.,0.,1.0);
        pangolin::glDrawFrustrum(cam.GetKinv(), w, h, T_wcRansac.matrix(), 0.1f);
      }
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,20, 0.03f);

      if (showFullPc) {
        pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
        pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
        // TODO I should not need to upload all of pc_w everytime;
        // might break things though
        // I do ned to upload points because they get updated; I
        // wouldnt have to with the color
        vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
        cbo_w.Upload(rgb_w.ptr_, rgb_w.SizeBytes(), 0);
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
          for (size_t i=0; i<assocBA.size(); ++i) {
            if (assocBA[i] > 0) {
              tdp::Vector3fda pA = T_wc*featsA[i].p_c_;
              tdp::glDrawLine(featsB[assocBA[i]].p_c_, pA);
            }
          }
        }
      }

      if (showNormals) {
        tdp::ShowCurrentNormals(pc, n, assoc, T_wc, scale);
        tdp::ShowGlobalNormals(pc_w, n_w, scale, step);
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
      nbo_w.Upload(n_w.ptr_, n_w.SizeBytes(), 0);
      pangolin::RenderVbo(nbo_w);
      glColor4f(1,0,0,1.);
      for (size_t k=0; k<dpvmf.GetK(); ++k) {
        tdp::glDrawLine(tdp::Vector3fda::Zero(), dpvmf.GetCenter(k));
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
      if (showFAST) {
        glColor3f(0,1,0);
        for (size_t i=0; i<pts.Area(); ++i) {
          pangolin::glDrawCircle(pts[i](0), pts[i](1), 1);
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

