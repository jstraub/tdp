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
#include <tdp/gl/gl_draw.h>
#include <tdp/gui/gui_base.hpp>
#include <tdp/gui/quickView.h>
#include <tdp/icp/icp.h>
#include <tdp/manifold/SE3.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/preproc/convolutionSeparable.h>
#include <tdp/preproc/depth.h>
#include <tdp/preproc/normals.h>
#include <tdp/preproc/pc.h>
#include <tdp/tsdf/tsdf.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/manifold/SO3.h>
#include <tdp/preproc/grad.h>
#include <tdp/preproc/grey.h>
#include <tdp/slam/keyframe.h>
#include <tdp/slam/keyframe_slam.h>
#include <tdp/gl/shaders.h>
#include <tdp/rtmf/vMFMMF.h>
#include <tdp/utils/colorMap.h>
#include <tdp/marching_cubes/marching_cubes.h>

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

int main( int argc, char* argv[] )
{
  std::string input_uri = "openni2://";
  std::string output_uri = "pango://video.pango";
  std::string calibPath = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";
  bool runOnce = false;

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
    if (argc > 3 && std::string(argv[3]).compare("-1") == 0 ) runOnce = true;
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


  Stopwatch::getInstance().setCustomSignature(1243984912);

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

  size_t dTSDF = 512;
  size_t wTSDF = 512;
  size_t hTSDF = 512;

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,false);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
  } else {
    return 2;
  }

  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = (w+w%64); // for convolution
  size_t hc = rig.NumCams()*(h+h%64);
  wc += wc%64;
  hc += hc%64;

  tdp::Camera<float> camView(Eigen::Vector4f(220,220,319.5,239.5)); 
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);

  pangolin::OpenGlRenderState camNormals(
      pangolin::ProjectionMatrix(640,3*480,420,3*420,320,3*240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  pangolin::View& viewNormals = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(camNormals));
  gui.container().AddDisplay(viewNormals);

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewKf(wc, hc);
  containerTracking.AddDisplay(viewKf);
  tdp::QuickView viewCurrent(wc, hc);
  containerTracking.AddDisplay(viewCurrent);
  gui.container().AddDisplay(containerTracking);


  pangolin::View& containerLoopClosure = pangolin::Display("loopClosures");
  containerLoopClosure.SetLayout(pangolin::LayoutEqual);
  pangolin::OpenGlRenderState camLoopClose(
      pangolin::ProjectionMatrix(640,3*480,420,3*420,320,3*240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  pangolin::View& viewLoopClose = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(camLoopClose));
  containerLoopClosure.AddDisplay(viewLoopClose);
  tdp::QuickView viewDebugA(wc, hc);
  containerLoopClosure.AddDisplay(viewDebugA);
  tdp::QuickView viewDebugB(wc, hc);
  containerLoopClosure.AddDisplay(viewDebugB);
  tdp::QuickView viewDebugC(wc, hc);
  containerLoopClosure.AddDisplay(viewDebugC);
  tdp::QuickView viewDebugD(wc, hc);
  containerLoopClosure.AddDisplay(viewDebugD);

  tdp::QuickView viewDebugE(wc, hc);
  containerLoopClosure.AddDisplay(viewDebugE);
  tdp::QuickView viewDebugF(wc, hc);
  containerLoopClosure.AddDisplay(viewDebugF);

  gui.container().AddDisplay(containerLoopClosure);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);
  tdp::ManagedDeviceImage<float> cuGrey(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreydv(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreydu(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuGrad3D(wc,hc);

  tdp::ManagedHostImage<float> greyA(wc, hc);
  tdp::ManagedHostImage<float> greyB(wc, hc);
  tdp::ManagedHostImage<float> photoErrBefore(wc, hc);
  tdp::ManagedHostImage<float> photoErrAfter(wc, hc);
  tdp::ManagedDeviceImage<float> cuPhotoErrAfter(wc, hc);

  tdp::SE3f T_abSuccess;

  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcA(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNA(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcB(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNB(wc, hc);
  tdp::ManagedHostImage<int> assoc_ba(wc,hc);
  tdp::ManagedDeviceImage<int> cuAssoc_ba(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedHostVolume<tdp::TSDFval> TSDF(wTSDF, hTSDF, dTSDF);
  TSDF.Fill(tdp::TSDFval(-1.01,0.));
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTSDF(wTSDF, hTSDF, dTSDF);
  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);

  // mesh buffers
  pangolin::GlBuffer meshVbo;
  pangolin::GlBuffer meshCbo;
  pangolin::GlBuffer meshIbo;

  tdp::ManagedDeviceImage<float> cuDView(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcView(wc, hc);
  
  tdp::ManagedHostPyramid<float,3> pyrGrey(wc,hc);

  // ICP stuff
  tdp::ManagedHostPyramid<float,3> dPyr(wc,hc);
  tdp::ManagedHostPyramid<float,3> dPyrEst(wc,hc);
  tdp::ManagedDevicePyramid<float,3> cuDPyr(wc,hc);
  tdp::ManagedDevicePyramid<float,3> cuDPyrEst(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> gs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> gs_c(wc,hc);

  tdp::ManagedDeviceImage<tdp::Vector3fda> nEstdummy(wc,hc);

  pangolin::GlBufferCudaPtr cuPcbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer vboSuccessA(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer vboSuccessB(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

  tdp::ManagedHostImage<float> tsdfDEst(wc, hc);
  tdp::ManagedHostImage<float> tsdfSlice(wTSDF, hTSDF);
//  tdp::QuickView viewTsdfDEst(wc,hc);
//  gui.container().AddDisplay(viewTsdfDEst);

  tdp::ManagedHostImage<float> dispDepthPyr(dPyr.Width(0)+dPyr.Width(1), hc);
  
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuDispNormalsPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuDispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedHostImage<tdp::Vector3bda> dispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);

  tdp::QuickView viewGrad3DPyr(dispNormals2dPyr.w_,dispNormals2dPyr.h_);
  gui.container().AddDisplay(viewGrad3DPyr);
  viewGrad3DPyr.Show(false);

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",6.,0.1,10.);
  pangolin::Var<bool> useRgbCamParasForDepth("ui.use RGB cam", true, true);

  pangolin::Var<bool> dispNormalsPyrEst("ui.disp normal est", false, true);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool> runSlamFusion("ui.run SLAM fusion", false,true);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<bool>  recomputeBoundingBox("ui.compute BB", false, false);

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  pangolin::Var<bool>  icpGrad3D("ui.run ICP Grad3D", false, true);
  pangolin::Var<float> gradNormThr("ui.grad3d norm thr",6.,0.,10.);

  pangolin::Var<bool> useOptimizedPoses("ui.use opt poses", true,true);
  pangolin::Var<bool> tryLoopClose("ui.loop close", true,true);
  pangolin::Var<bool> retryAllLoopClosures("ui.retry all loop close", false,false);
  pangolin::Var<bool> retryLoopClosure("ui.retry loop close", false,false);
  pangolin::Var<int>   loopCloseA("ui.loopClose A",0,0,10);
  pangolin::Var<int>   loopCloseB("ui.loopClose B",1,0,10);
  pangolin::Var<int>   numLoopClose("ui.Num loopClose",0,0,0);

  pangolin::Var<bool> useANN("ui.use ANN", false,true);
  pangolin::Var<bool> showAfterOpt("ui.show after opt", false,true);
  pangolin::Var<float> keyFrameDistThresh("ui.KF dist thr", 0.10, 0.01, 0.5);
  pangolin::Var<float> keyFrameAngleThresh("ui.KF angle thr", 10, 1, 50);
  pangolin::Var<int>  icpDownSample("ui.ICP downsample",100,1,200);
  pangolin::Var<float> loopCloseDistThresh( "ui.loop dist thr", 0.80, 0.01, 0.5);
  pangolin::Var<float> loopCloseAngleThresh("ui.loop angle thr", 90, 1, 180);
  pangolin::Var<float> icpLoopCloseAngleThr_deg("ui.icpLoop angle thr",20,0.,90.);
  pangolin::Var<float> icpLoopCloseDistThr("ui.icpLoop dist thr",0.30,0.,1.);
  pangolin::Var<int>   icpLoopCloseIter0("ui.icpLoop iter lvl 0",30,0,30);
  pangolin::Var<int>   icpLoopCloseOverlapLvl("ui.overlap lvl",0,0,2);
  pangolin::Var<float> icpLoopCloseOverlapThr("ui.overlap thr",0.30,0.,1.);
  pangolin::Var<float> rmseChangeThr("ui.dRMSE thr", 0.01,-1.,1.);
  pangolin::Var<float> rmseThr("ui.RMSE thr", 0.11,0.,1.);
  pangolin::Var<float> icpLoopCloseErrThr("ui.err thr",0.001,0.001,0.1);

  pangolin::Var<bool> runKfOnlyFusion("ui.run KF Fusion",true,false);

  pangolin::Var<bool> computePhotometricError("ui.comp Phot Err",true,true);

  pangolin::Var<bool>  runMarchingCubes("ui.run Marching Cubes", false, false);
  pangolin::Var<float> marchCubesfThr("ui.f Thr", 0.5,0.,1.);
  pangolin::Var<float> marchCubeswThr("ui.weight Thr", 0,0,10);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool>  saveTSDF("ui.save TSDF", false, false);
  pangolin::Var<float> tsdfMu("ui.mu",0.5,0.,1.);
  pangolin::Var<float> tsdfWThr("ui.w thr",25.,1.,20.);
  pangolin::Var<float> tsdfWMax("ui.w max",200.,1.,300.);
  pangolin::Var<float> grid0x("ui.grid0 x",-3.0,-2.,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-3.0,-2.,0);
  pangolin::Var<float> grid0z("ui.grid0 z",-3.0,-2.,0);
  pangolin::Var<float> gridEx("ui.gridE x", 3.0,2,3);
  pangolin::Var<float> gridEy("ui.gridE y", 3.0,2,3);
  pangolin::Var<float> gridEz("ui.gridE z", 3.0,2,3);
  pangolin::Var<int>   mmfId("ui.MMF id",0,0,2);

  pangolin::Var<bool> showPcModel("ui.show model",true,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",true,true);

  pangolin::Var<bool>  icpRot("ui.run ICP Rot", false, true);
  pangolin::Var<bool>  icpImu("ui.use IMU to warm start ICP", false, true);
  pangolin::Var<bool>  icpRotOverwrites("ui.ICP Rot Overwrites", false, true);
  pangolin::Var<int>   icpRotIter0("ui.ICP rot iter lvl 0",10,0,10);
  pangolin::Var<int>   icpRotIter1("ui.ICP rot iter lvl 1",7,0,10);
  pangolin::Var<int>   icpRotIter2("ui.ICP rot iter lvl 2",5,0,10);

  tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
  tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
  tdp::Vector3fda dGrid = gridE - grid0;

  tdp::ThreadedValue<bool> runSave(true);
  std::thread workThread([&]() {
        while(runSave.Get()) {
          if (pangolin::Pushed(saveTSDF)) {
            TSDF.CopyFrom(cuTSDF, cudaMemcpyDeviceToHost);
            std::cout << "start writing TSDF to " << tsdfOutputPath << std::endl;
            tdp::TSDF::SaveTSDF(TSDF, grid0, dGrid, tsdfOutputPath);
            std::cout << "done writing TSDF to " << tsdfOutputPath << std::endl;
          }
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      });

  tdp::SE3f T_mo_0;
  tdp::SE3f T_mo = T_mo_0;

  tdp::SE3f T_ac; // current to active KF

  tdp::KeyframeSLAM kfSLAM;
  std::vector<tdp::KeyFrame> kfs;
  std::vector<tdp::SE3f> T_mos;

  std::list<std::pair<int,int>> loopClose;
  std::map<std::pair<int,int>, float> rmses;

  gui.verbose = false;

  int idActive = 0;

  tdp::ThreadedValue<bool> runLoopClosure(true);
  std::mutex mut;

  auto computeLoopClosures = [&]() {
    size_t numLoopClosures = 0;
    size_t I = loopClose.size()/10 +1;
    for(size_t i=0; i < I; ++i) {
      std::pair<int,int> ids(-1,-1);
      if (loopClose.size() > 0) {
        ids = loopClose.front();
        loopClose.pop_front();
      }
      if (ids.first < 0 && ids.second < 0) {
        break;
      }
      tdp::KeyFrame& kfA = kfs[ids.first];
      tdp::KeyFrame& kfB = kfs[ids.second];
      Eigen::Matrix<float,6,1> se3 = kfA.T_wk_.Log(kfB.T_wk_);
      if ( se3.head<3>().norm()*180./M_PI < loopCloseAngleThresh
        && se3.tail<3>().norm()           < loopCloseDistThresh) {

        tdp::SE3f T_ab = kfA.T_wk_.Inverse() * kfB.T_wk_;
        std::cout << " checking " << ids.first << " -> " << ids.second 
          << ": " << se3.head<3>().norm()*180./M_PI << " "
          << se3.tail<3>().norm()          
          << std::endl;

        photoErrBefore.Fill(0.);
        float overlapBefore, rmseBefore;

        TICK("Overlap");
        cudaMemset(cuPhotoErrAfter.ptr_, 0, cuPhotoErrAfter.SizeBytes());
        Overlap(kfA, kfB, rig, icpLoopCloseOverlapLvl, overlapBefore, 
            rmseBefore, nullptr, &cuPhotoErrAfter);
        photoErrBefore.CopyFrom(cuPhotoErrAfter, cudaMemcpyDeviceToHost);
        TOCK("Overlap");

        if (overlapBefore > icpLoopCloseOverlapThr) {

          TICK("LoopClosure");
          float err=0.;
          float count=10000;
          if (useANN) {
            cuPcA.CopyFrom(kfA.pc_, cudaMemcpyHostToDevice);
            cuNA.CopyFrom(kfA.n_, cudaMemcpyHostToDevice);
            cuPcB.CopyFrom(kfB.pc_, cudaMemcpyHostToDevice);
            cuNB.CopyFrom(kfB.n_, cudaMemcpyHostToDevice);
            tdp::ICP::ComputeANN(kfA.pc_, cuPcA, cuNA, kfB.pc_, cuPcB, cuNB, 
              assoc_ba, cuAssoc_ba, T_ab, icpLoopCloseIter0, 
              icpLoopCloseAngleThr_deg, icpLoopCloseDistThr, 
              icpDownSample, gui.verbose, err, count);
            count *= icpDownSample;
          } else {
            // TODO test
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);

            pcs_m.CopyFrom(kfA.pyrPc_, cudaMemcpyHostToDevice);
            ns_m.CopyFrom(kfA.pyrN_, cudaMemcpyHostToDevice);
            pcs_c.CopyFrom(kfB.pyrPc_, cudaMemcpyHostToDevice);
            ns_c.CopyFrom(kfB.pyrN_, cudaMemcpyHostToDevice);

            std::vector<size_t> maxIt = {icpIter0, icpIter1, icpIter2};
//            tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, 
//                ns_c, T_ab, tdp::SE3f(),
//              camD, maxIt, icpLoopCloseAngleThr_deg,
//              icpLoopCloseDistThr, true | gui.verbose); 
            std::vector<float> errPerLvl;
            std::vector<float> countPerLvl;
            if (useRgbCamParasForDepth) {
              tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_c, ns_c,
                  rig, rig.rgbStream2cam_, maxIt, icpLoopCloseAngleThr_deg, 
                  icpLoopCloseDistThr,
                  gui.verbose, T_ab, errPerLvl, countPerLvl);
            } else {
              tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_c, ns_c,
                  rig, rig.dStream2cam_, maxIt, icpLoopCloseAngleThr_deg, 
                  icpLoopCloseDistThr,
                  gui.verbose, T_ab, errPerLvl, countPerLvl);
            }
            count = countPerLvl[0];
            err = errPerLvl[0];
          }
          TOCK("LoopClosure");

//          photoErrAfter.Fill(0.);
          cudaMemset(cuPhotoErrAfter.ptr_, 0, cuPhotoErrAfter.SizeBytes());
          float overlapAfter, rmseAfter;
          Overlap(kfA, kfB, rig, icpLoopCloseOverlapLvl,
              overlapAfter, rmseAfter, &T_ab, &cuPhotoErrAfter);
          photoErrAfter.CopyFrom(cuPhotoErrAfter, cudaMemcpyDeviceToHost);

          std::cout << "Overlap " << overlapBefore << " -> " << overlapAfter 
            << " RMSE " << rmseBefore << " -> " << rmseAfter 
            << " dRMSE " << (rmseBefore-rmseAfter)/rmseBefore
            << std::endl;

          if (err == err 
              && err < icpLoopCloseErrThr
              && count > 3000 
              && overlapAfter > icpLoopCloseOverlapThr
              && (rmseBefore-rmseAfter)/rmseBefore > rmseChangeThr
              && rmseAfter < rmseThr) {
            std::cout << GREEN << "successfull loop closure " 
              << NORMAL << std::endl
              << T_ab.matrix3x4() << std::endl;
            kfSLAM.AddLoopClosure(ids.first, ids.second, T_ab);
//            kfSLAM.AddLoopClosure(ids.second, ids.first, T_ab.Inverse());
//            loopClosures.emplace(std::make_pair(ids.first, ids.second), T_ab);
            numLoopClosures ++;

            // update views
            viewDebugA.SetImage(photoErrBefore);
            viewDebugB.SetImage(photoErrAfter);
            viewDebugC.SetImage(kfA.pyrGrey_.GetImage(0));
            viewDebugD.SetImage(kfB.pyrGrey_.GetImage(0));
            vboSuccessA.Upload(kfA.pc_.ptr_,kfA.pc_.SizeBytes(), 0);
            vboSuccessB.Upload(kfB.pc_.ptr_,kfB.pc_.SizeBytes(), 0);

            T_abSuccess = T_ab;

            std::cout << "optimizing graph" << std::endl;
            kfSLAM.Optimize(); 
            if (useOptimizedPoses) {
              T_ac = kfSLAM.GetPose(idActive).Inverse()*kfs[idActive].T_wk_*T_ac;
              for (size_t i=0; i < kfs.size(); ++i) {
                kfs[i].T_wk_ = kfSLAM.GetPose(i);
              }
            }

            if (computePhotometricError) {
              for (auto& it : kfSLAM.loopClosures_) {
                tdp::KeyFrame& kfA = kfs[it.first];
                tdp::KeyFrame& kfB = kfs[it.second];
                float overlap, rmse;
                TICK("Overlap");
                //        cudaMemset(cuPhotoErrAfter.ptr_, 0, cuPhotoErrAfter.SizeBytes());
                Overlap(kfA, kfB, rig, icpLoopCloseOverlapLvl, overlap, 
                    rmse, nullptr, nullptr); //&cuPhotoErrAfter);
                //        photoErrBefore.CopyFrom(cuPhotoErrAfter, cudaMemcpyDeviceToHost);
                TOCK("Overlap");
                std::cout << it.first << " to " << it.second << ":\tRMSE " << rmse
                  << "\toverlap " << overlap << std::endl;
                rmses[it] = rmse;
              }

              auto mapComp = [](const std::pair<std::pair<int,int>,float>& a, 
                  const std::pair<std::pair<int,int>,float>& b) -> bool {
                return a.second < b.second; };
              std::pair<int,int> idMax =
                std::max_element(rmses.begin(), rmses.end(),
                    mapComp)->first;

              viewDebugE.SetImage(kfs[idMax.first].pyrGrey_.GetImage(0));
              viewDebugF.SetImage(kfs[idMax.second].pyrGrey_.GetImage(0));
            }
            break;

          } else {
            std::cout << "unsuccessfull loop closure" << std::endl;
          }
        } else {
          std::cout << "aborting loop closure because overlap " << overlapBefore
            << " is to small" << std::endl;
        }
      } else {
        std::cout << " skipping " << ids.first << " -> " << ids.second  
          << ": " << se3.head<3>().norm()*180./M_PI << " "
          << se3.tail<3>().norm()          << std::endl;
      }
    }
  };

  std::thread loopClosureThread([&]() {
      while (runLoopClosure.Get()) {
//        computeLoopClosures();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      });

  if (gui.verbose) std::cout << "starting main loop" << std::endl;


  tdp::vMFMMF<3> mmf(10.);
  size_t Nmmf = 1000000;
  tdp::ManagedHostImage<tdp::Vector3fda> nMmf(Nmmf,1);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNMmf(Nmmf,1);
  pangolin::GlBuffer vboNMmf(pangolin::GlArrayBuffer,Nmmf,GL_FLOAT,3);

  tdp::SE3f T_wG;  // from grid to world
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    grid0 = tdp::Vector3fda(grid0x,grid0y,grid0z);
    gridE = tdp::Vector3fda(gridEx,gridEy,gridEz);
    dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    numLoopClose = loopClose.size();

    if ((runSlamFusion.GuiChanged() && runSlamFusion)
       || (gui.finished() && !runSlamFusion && loopClose.size() == 0)) {
      T_mos.clear();
      T_mo = kfs[0].T_wk_;
      idActive = 0;
      gui.Seek(0);
      gui.finished_ = false;
      resetTSDF = true;
      runSlamFusion = true;
      recomputeBoundingBox = true;
    }

    if (pangolin::Pushed(recomputeBoundingBox)) {

      for (size_t i=0; i<Nmmf; ++i) {
        tdp::Vector3fda ni ;
        do {
          Eigen::Vector2f ids = 
            0.5*(Eigen::Vector2f::Random()+Eigen::Vector2f::Ones());
          int32_t idKf = floor(ids(0)*kfs.size());
          int32_t idPt = floor(ids(1)*kfs[idKf].pyrN_.GetImage(2).Area());
          ni = kfs[idKf].T_wk_.rotation()*kfs[idKf].pyrN_.GetImage(2)[idPt];
        } while (!tdp::IsValidData(ni));
        nMmf[i] = ni;
      }
      cuNMmf.CopyFrom(nMmf, cudaMemcpyHostToDevice);
      mmf.Compute(cuNMmf, 10, true);
      size_t idMax = std::distance(mmf.Ns_.begin(),
          std::max_element(mmf.Ns_.begin(), mmf.Ns_.end()));
      std::cout << "largest MF: " << mmf.Ns_[idMax] << std::endl;
      std::cout << mmf.Rs_[idMax] << std::endl;
      for (size_t k=0; k<3; ++k) {
        std::cout << mmf.Rs_[k] << std::endl;
      }
      mmfId = idMax;
      T_wG.rotation() = mmf.Rs_[mmfId];

      grid0.fill(1e9);
      gridE.fill(-1e9);
      for (size_t i=0; i<Nmmf; ++i) {
        tdp::Vector3fda pi;
        do {
          Eigen::Vector2f ids = 
            0.5*(Eigen::Vector2f::Random()+Eigen::Vector2f::Ones());
          int32_t idKf = floor(ids(0)*kfs.size());
          int32_t idPt = floor(ids(1)*kfs[idKf].pyrPc_.GetImage(2).Area());
          pi = T_wG.Inverse()*kfs[idKf].T_wk_*kfs[idKf].pyrPc_.GetImage(2)[idPt];
        } while (!tdp::IsValidData(pi));
        grid0 = grid0.array().min(pi.array());
        gridE = gridE.array().max(pi.array());
      }
      grid0x = grid0(0);
      grid0y = grid0(1);
      grid0z = grid0(2);
      gridEx = gridE(0);
      gridEy = gridE(1);
      gridEz = gridE(2);
      dGrid = gridE - grid0;
      dGrid(0) /= (wTSDF-1);
      dGrid(1) /= (hTSDF-1);
      dGrid(2) /= (dTSDF-1);
      vboNMmf.Upload(nMmf.ptr_,nMmf.SizeBytes(), 0);
    }


    if (pangolin::Pushed(resetTSDF)) {
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      cuTSDF.CopyFrom(TSDF, cudaMemcpyHostToDevice);
      std::cout << "resetting TSDF" << std::endl;
    }

    if (kfs.size() > 1 && pangolin::Pushed(retryAllLoopClosures)) {
      for (int i=0; i<kfs.size()-1; ++i) 
        for (int j=i+1; j<kfs.size()-1; ++j) 
          loopClose.emplace_back(i,j);
    }

    if (loopCloseA < kfs.size() && loopCloseB < kfs.size() 
        && pangolin::Pushed(retryLoopClosure)) {
        loopClose.emplace_front(loopCloseA,loopCloseB);
    }

    if (pangolin::Pushed(runKfOnlyFusion)) {
      for (size_t i=0; i<kfs.size(); ++i) {
        if (true || gui.verbose)
          std::cout << "add KF " << i << " to tsdf" << std::endl;
        const auto& kfA = kfs[i];
        const tdp::SE3f& T_mk = kfA.T_wk_;
        cuD.CopyFrom(kfA.d_, cudaMemcpyHostToDevice);
        TICK("Add To TSDF");
//        AddToTSDF(cuTSDF, cuD, T_mk, camD, grid0, dGrid, tsdfMu, tsdfWMax); 
        rig.AddToTSDF(cuD, T_wG.Inverse()*T_mk, useRgbCamParasForDepth, 
            grid0, dGrid, tsdfMu, tsdfWMax, cuTSDF);
        TOCK("Add To TSDF");
      }
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();

    if (gui.frame < 1) continue;

    int64_t t_host_us_d = 0;
    if (gui.verbose) std::cout << "setup pyramids" << std::endl;
    TICK("Setup Pyramids");
    if (gui.verbose) std::cout << "collect d" << std::endl;
    rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
    if (gui.verbose) std::cout << "compute pc" << std::endl;
    rig.ComputePc(cuD, true, pcs_c);
    if (gui.verbose) std::cout << "compute n" << std::endl;
    rig.ComputeNormals(cuD, true, ns_c);
    if (gui.verbose) std::cout << "collect rgb" << std::endl;
    rig.CollectRGB(gui, rgb, cudaMemcpyHostToHost) ;
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGrey, 1./255.);
//    tdp::Image<tdp::Vector3fda> cuNs = ns_c.GetImage(0);
//    tdp::Image<tdp::Vector3fda> cuGs = gs_c.GetImage(0);
//    tdp::Gradient3D(cuGrey, cuD, cuNs, camD, gradNormThr, cuGreydu,
//        cuGreydv, cuGs);
//    tdp::CompletePyramid(gs_c, cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

    if(kfs.size() > 0 && !gui.finished()) {
      // Find closest KF
      int iMin = 0;
//      float distMin = 1e9;
//      float angMin = 1e9;
      float valMin = 1e9;
      for (int i=0; i<kfs.size(); ++i) {
        Eigen::Matrix<float,6,1> se3 = kfs[i].T_wk_.Log(T_mo);
        float dist = se3.tail<3>().norm();
        float ang = se3.head<3>().norm();
//        if (ang < angMin && dist < distMin) {
//          distMin = dist;
//          angMin = ang;
        if (2*ang+dist < valMin) {
          valMin = 2*ang+dist;
          iMin = i;
        }
      }
      if (iMin != idActive) {
        std::cout << "switching to tracking against KF " << iMin << std::endl;
        T_ac = kfs[iMin].T_wk_.Inverse()*kfs[idActive].T_wk_*T_ac;
        std::cout << T_ac << std::endl;
        idActive = iMin;
        viewKf.SetImage(kfs[idActive].rgb_);
      } 

      tdp::KeyFrame& kf = kfs[idActive];
      pcs_m.CopyFrom(kf.pyrPc_,cudaMemcpyHostToDevice);
      ns_m.CopyFrom(kf.pyrN_,cudaMemcpyHostToDevice);
      // TODO:
      //      gs_m.CopyFrom(gs_c,cudaMemcpyDeviceToDevice);

      if (gui.verbose) std::cout << "icp" << std::endl;
      TICK("ICP");
      std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
      if (!icpGrad3D) {
//        tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, T_ac, tdp::SE3f(),
//            camD, maxIt, icpAngleThr_deg, icpDistThr, gui.verbose); 
        std::vector<float> errPerLvl;
        std::vector<float> countPerLvl;
        if (useRgbCamParasForDepth) {
          tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_c, ns_c,
              rig, rig.rgbStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
              gui.verbose, T_ac, errPerLvl, countPerLvl);
        } else {
          tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_c, ns_c,
              rig, rig.dStream2cam_, maxIt, icpAngleThr_deg, 
              icpDistThr,
              gui.verbose, T_ac, errPerLvl, countPerLvl);
        }
//      } else {
//        tdp::ICP::ComputeProjective(pcs_m, ns_m, gs_m, pcs_c, ns_c,
//            gs_c, T_ac, tdp::SE3f(), camD, maxIt, icpAngleThr_deg,
//            icpDistThr, gui.verbose); 
      }
      TOCK("ICP");
      T_mo = kfs[idActive].T_wk_*T_ac;
      T_mos.push_back(T_mo);
    }

    if (!runSlamFusion) {
      Eigen::Matrix<float,6,1> se3 = Eigen::Matrix<float,6,1>::Zero();
      if (kfs.size() > 0) 
        se3 = kfs[idActive].T_wk_.Log(T_mo);
      if ( (kfs.size() == 0)
          || se3.head<3>().norm()*180./M_PI > keyFrameAngleThresh
          || se3.tail<3>().norm() > keyFrameDistThresh) {
        std::cout << "adding keyframe " << kfs.size() 
          << " angle: " << se3.head<3>().norm()*180./M_PI 
          << " dist: " << se3.tail<3>().norm() 
          << " T_mk: " << std::endl << T_mo
          << std::endl;

        tdp::ConstructPyramidFromImage(cuGrey, pyrGrey, cudaMemcpyDeviceToHost);
        kfs.emplace_back(pcs_c, ns_c, pyrGrey, rgb, cuD, T_mo);
        if (kfs.size() == 1) {
          std::cout << "first KF -> adding origin" << std::endl;
          kfSLAM.AddOrigin(T_mo);
        } else {
          std::cout << "not first KF -> adding ICP odom "
            << kfs.size()-1 << " to " << idActive 
            << std::endl;
          kfSLAM.AddIcpOdometry(idActive, kfs.size()-1, T_ac);
        }

        for (int i=0; i<kfs.size()-1; ++i) {
          loopClose.emplace_front(kfs.size()-1,i);
        }
        idActive = kfs.size()-1;
        T_ac = tdp::SE3f();
        T_mo = kfs[idActive].T_wk_*T_ac;
        viewKf.SetImage(kfs[idActive].rgb_);
      }
      if (tryLoopClose) {
        computeLoopClosures();
      }
    } else {
      if (!gui.finished()) {
        TICK("Add To TSDF");
        rig.AddToTSDF(cuD, T_wG.Inverse()*T_mo, useRgbCamParasForDepth, 
            grid0, dGrid, tsdfMu, tsdfWMax, cuTSDF);
        TOCK("Add To TSDF");
      }
    }

    if (pangolin::Pushed(runMarchingCubes)
        || (runSlamFusion && gui.finished() && meshVbo.num_elements == 0)) {
      TSDF.CopyFrom(cuTSDF, cudaMemcpyDeviceToHost);
      tdp::ComputeMesh(TSDF, grid0, dGrid,
          T_wG, meshVbo, meshCbo, meshIbo, marchCubeswThr, marchCubesfThr);      
      saveTSDF = true;
      if (runOnce) break;
    }

    if (gui.verbose) std::cout << "draw 3D" << std::endl;

    TICK("Draw 3D");

    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      T_wG.rotation() = tdp::SO3f(mmf.Rs_[mmfId]);
      pangolin::glSetFrameOfReference(T_wG.matrix());
      Eigen::AlignedBox3f box(grid0,gridE);
      glColor4f(1,0,0,0.5f);
      pangolin::glDrawAlignedBox(box);
      pangolin::glUnsetFrameOfReference();

      pangolin::glDrawAxis(kfs[idActive].T_wk_.matrix(),0.08f);
      pangolin::glDrawAxis(T_mo.matrix(), 0.05f);
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_mos,-1);
      for (size_t i=0; i<kfs.size(); ++i) {
        tdp::SE3f& T_wk = kfs[i].T_wk_;
        pangolin::glDrawAxis(T_wk.matrix(), 0.03f);
      }
      for (size_t i=0; i<kfSLAM.size(); ++i) {
        tdp::SE3f T_wk = kfSLAM.GetPose(i);
        pangolin::glDrawAxis(T_wk.matrix(), 0.03f);
      }

      if (loopCloseA < kfSLAM.size()) {
        tdp::SE3f T_wk = kfSLAM.GetPose(loopCloseA);
        pangolin::glDrawFrustrum(rig.cams_[0].GetKinv(), 640,480,
            T_wk.matrix(), 0.03f);
      }
      if (loopCloseB < kfSLAM.size()) {
        tdp::SE3f T_wk = kfSLAM.GetPose(loopCloseB);
        pangolin::glDrawFrustrum(rig.cams_[0].GetKinv(), 640,480,
            T_wk.matrix(), 0.03f);
      }

      if (!useOptimizedPoses) {
        glColor4f(1.,0.3,0.3,0.6);
        for (auto& it : kfSLAM.loopClosures_) {
          tdp::SE3f& T_wk_A = kfs[it.first].T_wk_;
          tdp::SE3f& T_wk_B = kfs[it.second].T_wk_;
          tdp::glDrawLine(T_wk_A.translation(), T_wk_B.translation());
        }
      }

      auto mapComp = [](const std::pair<std::pair<int,int>,float>& a, 
          const std::pair<std::pair<int,int>,float>& b) -> bool {
            return a.second < b.second; };
      float rmseMin = std::min_element(rmses.begin(), rmses.end(),
          mapComp)->second;
      float rmseMax = std::max_element(rmses.begin(), rmses.end(),
          mapComp)->second;
      for (auto& it : kfSLAM.loopClosures_) {
        tdp::SE3f T_wk_A = kfSLAM.GetPose(it.first);
        tdp::SE3f T_wk_B = kfSLAM.GetPose(it.second);
        if (rmses.find(it) != rmses.end()) {
          tdp::Vector3bda c = tdp::ColorMapHot(
              (rmses[it]-rmseMin)/(rmseMax-rmseMin));
          glColor4f(c(0)/255.,c(1)/255.,c(2)/255.,1.); 
        } else {
          glColor4f(0.,0.0,1.0,1.0); 
        }
        tdp::glDrawLine(T_wk_A.translation(), T_wk_B.translation());
      }
      // render model and observed point cloud
      if (showPcModel && kfs.size() > 0) {
        tdp::KeyFrame& kf = kfs[idActive];
        pcs_m.CopyFrom(kf.pyrPc_,cudaMemcpyHostToDevice);
        ns_m.CopyFrom(kf.pyrN_,cudaMemcpyHostToDevice);

        pangolin::glSetFrameOfReference(kfs[idActive].T_wk_.matrix());
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = pcs_m.GetImage(dispLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(0,1,0);
        pangolin::RenderVbo(cuPcbuf);
        pangolin::glUnsetFrameOfReference();
      }
      // render current camera second in the propper frame of
      // reference
      if (showPcCurrent) {
        pangolin::glSetFrameOfReference(T_mo.matrix());
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = pcs_c.GetImage(dispLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(1,0,0);
        pangolin::RenderVbo(cuPcbuf);
        pangolin::glUnsetFrameOfReference();
      }


      if (meshVbo.num_elements > 0
          && meshCbo.num_elements > 0
          && meshIbo.num_elements > 0) {
        meshVbo.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
        meshCbo.Bind();
        glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0); 
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        auto& shader = tdp::Shaders::Instance()->normalMeshShader_;   
        shader.Bind();
        pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
        pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
        shader.SetUniform("P",P);
        shader.SetUniform("MV",MV);

        meshIbo.Bind();
        glDrawElements(GL_TRIANGLES, meshIbo.num_elements*3,
            meshIbo.datatype, 0);
        meshIbo.Unbind();

        shader.Unbind();
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(0);
        meshCbo.Unbind();
        meshVbo.Unbind();
      }

    }

    if (viewNormals.IsShown()) {
      viewNormals.Activate(camNormals);

      pangolin::glDrawAxis(0.1f);
      glColor4f(1.f,0.f,0.f,0.5f);
      pangolin::RenderVbo(vboNMmf);

      for (size_t i=0; i<mmf.Rs_.size(); ++i) {
        tdp::SE3f T_wmmf(tdp::SO3f(mmf.Rs_[i]));
        pangolin::glDrawAxis(T_wmmf.matrix(),0.2f);
      }
    }

    if (viewLoopClose.IsShown() && kfs.size() > 1) {
      viewLoopClose.Activate(camLoopClose);

      pangolin::glDrawAxis(0.1f);
      glColor4f(1.f,0.f,0.f,0.5f);
      pangolin::RenderVbo(vboSuccessA);

      pangolin::glSetFrameOfReference(T_abSuccess.matrix());
      pangolin::glDrawAxis(0.1f);
      glColor4f(0.f,1.f,0.f,0.5f);
      pangolin::RenderVbo(vboSuccessB);
      pangolin::glUnsetFrameOfReference();

    }

    TOCK("Draw 3D");

    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (viewKf.IsShown()) {
      viewKf.RenderImage();
    }
    if (viewCurrent.IsShown()) {
      viewCurrent.SetImage(rgb);
    }
    if (viewDebugA.IsShown()) {
      viewDebugA.RenderImage();
    }
    if (viewDebugB.IsShown()) {
      viewDebugB.RenderImage();
    }
    if (viewDebugC.IsShown()) {
      viewDebugC.RenderImage();
    }
    if (viewDebugD.IsShown()) {
      viewDebugD.RenderImage();
    }
    if (viewDebugE.IsShown()) {
      viewDebugE.RenderImage();
    }
    if (viewDebugF.IsShown()) {
      viewDebugF.RenderImage();
    }

    if (viewGrad3DPyr.IsShown()) {
      tdp::PyramidToImage<tdp::Vector3fda,3>(gs_c,cuDispNormalsPyr,
          cudaMemcpyDeviceToDevice);
      tdp::RenormalizeSurfaceNormals(cuDispNormalsPyr, 1e-3);
      tdp::Normals2Image(cuDispNormalsPyr, cuDispNormals2dPyr);
      dispNormals2dPyr.CopyFrom(cuDispNormals2dPyr,cudaMemcpyDeviceToHost);
      viewGrad3DPyr.SetImage(dispNormals2dPyr);
    }

    TOCK("Draw 2D");

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }

  saveTSDF = true;

  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;

  std::this_thread::sleep_for(std::chrono::microseconds(500));

  runSave.Set(false);
  workThread.join();
  runLoopClosure.Set(false);
  loopClosureThread.join();
  return 0;
}

