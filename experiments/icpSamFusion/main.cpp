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

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

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
    imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  }
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

  Eigen::Matrix3f R_ir;
  R_ir << 0, 0,-1,
       0,-1, 0,
       -1, 0, 0;
  tdp::SE3f T_ir(R_ir,Eigen::Vector3f::Zero());

  tdp::GuiBase gui(1200,800,video);
  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = w+w%64; // for convolution
  size_t hc = h+h%64;
  float f = 550;
  float uc = (w-1.)/2.;
  float vc = (h-1.)/2.;

  size_t dTSDF = 512;
  size_t wTSDF = 512;
  size_t hTSDF = 512;

  CameraT camR(Eigen::Vector4f(f,f,uc,vc)); 
  CameraT camD(Eigen::Vector4f(f,f,uc,vc)); 

  if (calibPath.size() > 0) {
    tdp::Rig<CameraT> rig;
    rig.FromFile(calibPath,false);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
    // camera model for computing point cloud and normals
    camR = rig.cams_[rig.rgbStream2cam_[0]];
    camD = camR; //rig.cams_[rig.dStream2cam_[0]];
    if (rig.T_ris_.size() > 0) T_ir = rig.T_ris_[0];
  }

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

  pangolin::OpenGlRenderState camLoopClose(
      pangolin::ProjectionMatrix(640,3*480,420,3*420,320,3*240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  pangolin::View& viewLoopClose = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(camLoopClose));
  gui.container().AddDisplay(viewLoopClose);

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

  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcA(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNA(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcB(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNB(wc, hc);
  tdp::ManagedHostImage<int> assoc_ba(w,h);
  tdp::ManagedDeviceImage<int> cuAssoc_ba(w,h);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedHostVolume<tdp::TSDFval> TSDF(wTSDF, hTSDF, dTSDF);
  TSDF.Fill(tdp::TSDFval(-1.01,0.));
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTSDF(wTSDF, hTSDF, dTSDF);
  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);

  tdp::ManagedHostImage<float> dEst(wc, hc);
//  tdp::ManagedDeviceImage<float> cuDEst(wc, hc);
//  dEst.Fill(0.);
//  tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
  tdp::ManagedDeviceImage<float> cuDView(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcView(wc, hc);
  
  tdp::ManagedHostPyramid<float,3> pyrGrey(w,h);

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
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

  tdp::ManagedHostImage<float> tsdfDEst(wc, hc);
  tdp::ManagedHostImage<float> tsdfSlice(wTSDF, hTSDF);
//  tdp::QuickView viewTsdfDEst(wc,hc);
//  gui.container().AddDisplay(viewTsdfDEst);

  tdp::ManagedHostImage<float> dispDepthPyr(dPyr.Width(0)+dPyr.Width(1), hc);
  tdp::QuickView viewDepthPyr(dispDepthPyr.w_,dispDepthPyr.h_);
  gui.container().AddDisplay(viewDepthPyr);
  
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuDispNormalsPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuDispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedHostImage<tdp::Vector3bda> dispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);

  tdp::QuickView viewNormalsPyr(dispNormals2dPyr.w_,dispNormals2dPyr.h_);
  gui.container().AddDisplay(viewNormalsPyr);
  tdp::QuickView viewGrad3DPyr(dispNormals2dPyr.w_,dispNormals2dPyr.h_);
  gui.container().AddDisplay(viewGrad3DPyr);

  tdp::QuickView viewDebugA(wc, hc);
  gui.container().AddDisplay(viewDebugA);
  tdp::QuickView viewDebugB(wc, hc);
  gui.container().AddDisplay(viewDebugB);
  tdp::QuickView viewDebugC(wc, hc);
  gui.container().AddDisplay(viewDebugC);
  tdp::QuickView viewDebugD(wc, hc);
  gui.container().AddDisplay(viewDebugD);

  viewDebugA.Show(false);
  viewDebugB.Show(false);
  viewDebugC.Show(false);
  viewDebugD.Show(false);

  viewDepthPyr.Show(false);
  viewNormalsPyr.Show(false);
  viewGrad3DPyr.Show(false);

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> tsdfDmin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> tsdfDmax("ui.d max",6.,0.1,10.);

  pangolin::Var<bool> dispNormalsPyrEst("ui.disp normal est", false, true);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);


  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  pangolin::Var<bool>  icpGrad3D("ui.run ICP Grad3D", false, true);
  pangolin::Var<float> gradNormThr("ui.grad3d norm thr",6.,0.,10.);

  pangolin::Var<bool> tryLoopClose("ui.loop close", true,true);
  pangolin::Var<bool> useANN("ui.use ANN", true,true);
  pangolin::Var<bool> showAfterOpt("ui.show after opt", false,true);
  pangolin::Var<float> keyFrameDistThresh("ui.KF dist thr", 0.20, 0.01, 0.5);
  pangolin::Var<float> keyFrameAngleThresh("ui.KF angle thr", 10, 1, 50);
  pangolin::Var<int>  icpDownSample("ui.ICP downsample",300,1,100);
  pangolin::Var<float> icpLoopCloseAngleThr_deg("ui.icpLoop angle thr",20,0.,90.);
  pangolin::Var<float> icpLoopCloseDistThr("ui.icpLoop dist thr",0.30,0.,1.);
  pangolin::Var<int>   icpLoopCloseIter0("ui.icpLoop iter lvl 0",30,0,30);
  pangolin::Var<int>   icpLoopCloseOverlapLvl("ui.overlap lvl",0,0,2);
  pangolin::Var<float> icpLoopCloseOverlapThr("ui.overlap thr",0.50,0.,1.);
  pangolin::Var<float> icpLoopCloseErrThr("ui.err thr",0.001,0.001,0.1);

  pangolin::Var<bool> runFusion("ui.run Fusion",true,false);
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

  pangolin::Var<bool> showPcModel("ui.show model",true,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",true,true);

  pangolin::Var<bool>  icpRot("ui.run ICP Rot", false, true);
  pangolin::Var<bool>  icpImu("ui.use IMU to warm start ICP", false, true);
  pangolin::Var<bool>  icpRotOverwrites("ui.ICP Rot Overwrites", false, true);
  pangolin::Var<int>   icpRotIter0("ui.ICP rot iter lvl 0",10,0,10);
  pangolin::Var<int>   icpRotIter1("ui.ICP rot iter lvl 1",7,0,10);
  pangolin::Var<int>   icpRotIter2("ui.ICP rot iter lvl 2",5,0,10);

  tdp::ThreadedValue<bool> runSave(true);
  std::thread workThread([&]() {
        while(runSave.Get()) {
          if (pangolin::Pushed(saveTSDF)) {
            tdp::ManagedHostVolume<tdp::TSDFval> tmpTSDF(wTSDF, hTSDF, dTSDF);
            tmpTSDF.CopyFrom(cuTSDF, cudaMemcpyDeviceToHost);
            std::cout << "start writing TSDF to " << tsdfOutputPath << std::endl;
            tdp::SaveVolume(tmpTSDF, tsdfOutputPath);
            std::cout << "done writing TSDF to " << tsdfOutputPath << std::endl;
          }
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      });

  gui.verbose = true;

  tdp::SE3f T_mo_0;
  tdp::SE3f T_mo = T_mo_0;
  tdp::SE3f T_mo_prev = T_mo_0;

  tdp::SE3f T_ac; // current to active KF

  tdp::KeyframeSLAM kfSLAM;
  std::map<std::pair<int,int>,tdp::SE3f> loopClosures;
  std::vector<tdp::KeyFrame> kfs;
  std::vector<tdp::SE3f> T_mos;

  std::list<std::pair<int,int>> loopClose;

  gui.verbose = false;

  int idActive = 0;


  tdp::ThreadedValue<bool> runLoopClosure(true);
  std::mutex mut;

  auto computeLoopClosures = [&]() {
    int idA =-1;
    int idB =-1;
    size_t numLoopClosures = 0;
    {
      std::lock_guard<std::mutex> lock(mut);
      if (loopClose.size() > 0) {
        std::pair<int,int> ids = loopClose.front();
        loopClose.pop_front();
        idA = ids.first;
        idB = ids.second;
      }
//      std::cout << "In thread: # loop closure hypotheses " 
//        << loopClose.size() << " " << idA << " " << idB << std::endl;
    }
    if (idA > 0 && idB > 0) {
      tdp::KeyFrame& kfA = kfs[idA];
      cuPcA.CopyFrom(kfA.pc_, cudaMemcpyHostToDevice);
      cuNA.CopyFrom(kfA.n_, cudaMemcpyHostToDevice);
      tdp::KeyFrame& kfB = kfs[idB];
      Eigen::Matrix<float,6,1> se3 = kfA.T_wk_.Log(kfB.T_wk_);
      if ( se3.head<3>().norm()*180./M_PI < 1.5*keyFrameAngleThresh
        && se3.tail<3>().norm()           < 1.5*keyFrameDistThresh) {

        tdp::SE3f T_ab = kfA.T_wk_.Inverse() * kfB.T_wk_;
        std::cout << " checking " << idA << " -> " << idB 
          << ": " << se3.head<3>().norm()*180./M_PI << " "
          << se3.tail<3>().norm()          
          << std::endl;

        photoErrBefore.Fill(0.);
        float overlapBefore, rmseBefore;
        Overlap(kfA, kfB, camD, icpLoopCloseOverlapLvl, overlapBefore, 
            rmseBefore, nullptr, &photoErrBefore);

        if (overlapBefore > icpLoopCloseOverlapThr) {
          cuPcB.CopyFrom(kfB.pc_, cudaMemcpyHostToDevice);
          cuNB.CopyFrom(kfB.n_, cudaMemcpyHostToDevice);

          float err=0.;
          float count=10000;
          if (useANN) {
            tdp::ICP::ComputeANN(kfA.pc_, cuPcA, cuNA, kfB.pc_, cuPcB, cuNB, 
              assoc_ba, cuAssoc_ba, T_ab, icpLoopCloseIter0, 
              icpLoopCloseAngleThr_deg, icpLoopCloseDistThr, 
              icpDownSample, gui.verbose, err, count);
          } else {

            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
            tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);

            pcs_m.CopyFrom(kfA.pyrPc_, cudaMemcpyHostToDevice);
            ns_m.CopyFrom(kfA.pyrN_, cudaMemcpyHostToDevice);
            pcs_m.CopyFrom(kfB.pyrPc_, cudaMemcpyHostToDevice);
            ns_m.CopyFrom(kfB.pyrN_, cudaMemcpyHostToDevice);

            std::vector<size_t> maxIt = {2*icpIter0, 2*icpIter1, 2*icpIter2};
            tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, 
                ns_c, T_ab, tdp::SE3f(),
              camD, maxIt, icpLoopCloseAngleThr_deg,
              icpLoopCloseDistThr, true | gui.verbose); 
          }

          photoErrAfter.Fill(0.);
          float overlapAfter, rmseAfter;
          Overlap(kfA, kfB, camD, icpLoopCloseOverlapLvl,
              overlapAfter, rmseAfter, &T_ab, &photoErrAfter);

          greyA.CopyFrom(kfA.pyrGrey_.GetImage(0), cudaMemcpyHostToHost);
          greyB.CopyFrom(kfB.pyrGrey_.GetImage(0), cudaMemcpyHostToHost);

          std::cout << "Overlap " << overlapBefore << " -> " << overlapAfter 
            << " RMSE " << rmseBefore << " -> " << rmseAfter << std::endl;

          if (err == err 
              && err < icpLoopCloseErrThr
              && count*icpDownSample > 30000 
              && overlapAfter > icpLoopCloseOverlapThr
              && rmseAfter <= rmseBefore) {
            std::cout << "successfull loop closure "  << std::endl
              << T_ab.matrix3x4() << std::endl;
            kfSLAM.AddLoopClosure(idB, idA, T_ab.Inverse());
            loopClosures.emplace(std::make_pair(idA, idB), T_ab);
            numLoopClosures ++;
          }
        }
      } else {
        std::cout << " skipping " << idA << " -> " << idB  
          << ": "
          << se3.head<3>().norm()*180./M_PI << " "
          << se3.tail<3>().norm()          
          << std::endl;
      }
      if (numLoopClosures > 0) {
        kfSLAM.Optimize(); 
      }
    }
  };

  std::thread loopClosureThread([&]() {
      while (runLoopClosure.Get()) {
//        computeLoopClosures();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      });

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
    tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
    tdp::Vector3fda dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();
    tdp::Image<uint16_t> dRaw;
    int64_t t_host_us_d = 0;
    if (!gui.ImageD(dRaw,0,&t_host_us_d)) continue;
    if (gui.verbose) std::cout << "setup pyramids" << std::endl;
    TICK("Setup Pyramids");
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    ConvertDepthGpu(cuDraw, cuD, depthSensorScale, tsdfDmin, tsdfDmax);
    // construct pyramid  
    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
        cudaMemcpyDeviceToDevice, 0.03);
    tdp::Depth2PCsGpu(cuDPyr,camD,pcs_c);
    // compute normals
    tdp::Depth2Normals(cuDPyr,camD,ns_c);

    if (!gui.ImageRGB(rgb,0)) continue;
    cudaMemset(cuRgb.ptr_, 0, cuRgb.SizeBytes());
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGrey, 1./255.);
    tdp::Image<tdp::Vector3fda> cuNs = ns_c.GetImage(0);
    tdp::Image<tdp::Vector3fda> cuGs = gs_c.GetImage(0);
    tdp::Gradient3D(cuGrey, cuD, cuNs, camD, gradNormThr, cuGreydu,
        cuGreydv, cuGs);
    tdp::CompletePyramid(gs_c, cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

    if(kfs.size() > 0) {
      // Find closest KF
      int iMin = 0;
      float distMin = 1e9;
      float angMin = 1e9;
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
        idActive = iMin;
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
        tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, T_ac, tdp::SE3f(),
            camD, maxIt, icpAngleThr_deg, icpDistThr, gui.verbose); 
      } else {
        tdp::ICP::ComputeProjective(pcs_m, ns_m, gs_m, pcs_c, ns_c,
            gs_c, T_ac, tdp::SE3f(), camD, maxIt, icpAngleThr_deg,
            icpDistThr, gui.verbose); 
      }
      TOCK("ICP");
      T_mo = kfs[idActive].T_wk_*T_ac;
      T_mos.push_back(T_mo);

    }

    Eigen::Matrix<float,6,1> se3 = Eigen::Matrix<float,6,1>::Zero();
    if (kfs.size() > 0) 
      se3 = kfs[idActive].T_wk_.Log(T_mo);
    if ((kfs.size() == 0)
        || se3.head<3>().norm()*180./M_PI > keyFrameAngleThresh
        || se3.tail<3>().norm() > keyFrameDistThresh) {
      std::cout << "adding keyframe " << kfs.size() 
        << " angle: " << se3.head<3>().norm()*180./M_PI 
        << " dist: " << se3.tail<3>().norm() 
        << " T_mk: " << std::endl << T_mo
        << std::endl;
      pc.CopyFrom(pcs_c.GetImage(0),cudaMemcpyDeviceToHost);
      n.CopyFrom(ns_c.GetImage(0),cudaMemcpyDeviceToHost);
      kfSLAM.AddKeyframe(pc, n, rgb, T_mo);

      tdp::ConstructPyramidFromImage(cuGrey, pyrGrey, cudaMemcpyDeviceToHost);
      kfs.emplace_back(pcs_c, ns_c, pyrGrey, rgb, T_mo);

      if (kfs.size() > 1) {
        int idA = (int)kfs.size()-1;
        int idB = (int)kfs.size()-2;
        tdp::SE3f T_ab = kfs[idA].T_wk_.Inverse()*kfs[idB].T_wk_;
        loopClosures.emplace(std::make_pair(idA, idB), T_ab);
      }
      {
        std::lock_guard<std::mutex> lock(mut);
        for (int i=0; i<kfs.size()-1; ++i) {
          loopClose.emplace_back(kfs.size()-1,i);
        }
        std::cout << "# loop closure hypotheses " 
          << loopClose.size()<< std::endl;
      }
    } else {
      std::cout << "NOT adding keyframe " << kfs.size() 
        << " angle: " << se3.head<3>().norm()*180./M_PI 
        << " dist: " << se3.tail<3>().norm() 
        << " active: " << idActive << "/" << kfs.size()
        << std::endl;
    }

    if (tryLoopClose) {
      computeLoopClosures();
    }

    if (pangolin::Pushed(runFusion)) {
      // TODO: iterate over kfs and fuse them into TSDF
      for (size_t i=0; i<kfs.size(); ++i) {
        if (gui.verbose) std::cout << "add to tsdf" << std::endl;
        const auto& kfA = kfs[i];
        const tdp::SE3f& T_mk = kfA.T_wk_;
        cuD.CopyFrom(kfA.d_, cudaMemcpyHostToDevice);
        TICK("Add To TSDF");
        AddToTSDF(cuTSDF, cuD, T_mk, camD, grid0, dGrid, tsdfMu, tsdfWMax); 
        TOCK("Add To TSDF");
      }
//      if (gui.verbose) std::cout << "ray trace" << std::endl;
//      TICK("Ray Trace TSDF");
//      RayTraceTSDF(cuTSDF, pcs_m.GetImage(0), 
//          ns_m.GetImage(0), T_mo, camD, grid0, dGrid, tsdfMu, tsdfWThr); 
//      tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_m, cudaMemcpyDeviceToDevice);
//      tdp::CompleteNormalPyramid<3>(ns_m, cudaMemcpyDeviceToDevice);
//      TOCK("Ray Trace TSDF");
    }

    if (pangolin::Pushed(resetTSDF)) {
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      dEst.Fill(0.);
      cuTSDF.CopyFrom(TSDF, cudaMemcpyHostToDevice);
      T_mo = T_mo_0;
      T_mo_prev = T_mo;
      std::cout << "resetting ICP and TSDF" << std::endl;
    }
    if (pangolin::Pushed(resetICP)) {
      T_mo = T_mo_0;
      T_mo_prev = T_mo;
      std::cout << "resetting ICP" << std::endl;
    }

    if (gui.verbose) std::cout << "draw 3D" << std::endl;

    TICK("Draw 3D");

    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      Eigen::AlignedBox3f box(grid0,gridE);
      glColor4f(1,0,0,0.5f);
      pangolin::glDrawAlignedBox(box);

      pangolin::glDrawAxis(kfs[idActive].T_wk_.matrix(),0.08f);
      pangolin::glDrawAxis(T_mo.matrix(), 0.05f);
      for (size_t i=0; i<T_mos.size(); ++i) {
        glColor4f(1.,1.,0.,0.6);
        if (i>0) {
          pangolin::glDrawLine(
              T_mos[i].translation()(0), T_mos[i].translation()(1),
              T_mos[i].translation()(2),
              T_mos[i-1].translation()(0), T_mos[i-1].translation()(1),
              T_mos[i-1].translation()(2));
        }
      }
      for (size_t i=0; i<kfs.size(); ++i) {
        tdp::SE3f& T_wk = kfs[i].T_wk_;
        pangolin::glDrawAxis(T_wk.matrix(), 0.03f);
      }
      glColor4f(1.,0.3,0.3,0.6);
      for (auto& it : loopClosures) {
        tdp::SE3f& T_wk_A = kfs[it.first.first].T_wk_;
        tdp::SE3f& T_wk_B = kfs[it.first.second].T_wk_;
        pangolin::glDrawLine(
            T_wk_A.translation()(0), T_wk_A.translation()(1),
            T_wk_A.translation()(2),
            T_wk_B.translation()(0), T_wk_B.translation()(1),
            T_wk_B.translation()(2));
      }
      for (size_t i=0; i<kfSLAM.size(); ++i) {
        tdp::SE3f T_wk = kfSLAM.GetPose(i);
        pangolin::glDrawAxis(T_wk.matrix(), 0.03f);
      }
      glColor4f(0.,1.0,1.0,0.6);
      for (auto& it : loopClosures) {
        tdp::SE3f T_wk_A = kfSLAM.GetPose(it.first.first);
        tdp::SE3f T_wk_B = kfSLAM.GetPose(it.first.second);
        pangolin::glDrawLine(
            T_wk_A.translation()(0), T_wk_A.translation()(1),
            T_wk_A.translation()(2),
            T_wk_B.translation()(0), T_wk_B.translation()(1),
            T_wk_B.translation()(2));
      }
      // render model and observed point cloud
      if (showPcModel && kfs.size() > 0) {
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
    }

    if (viewLoopClose.IsShown() && kfs.size() > 1) {
      viewLoopClose.Activate(camLoopClose);
      tdp::KeyFrame& kfA = kfs[kfs.size()-1];
      tdp::KeyFrame& kfB = kfs[kfs.size()-2];

      tdp::SE3f T_ab = kfA.T_wk_.Inverse() * kfB.T_wk_;
      if (showAfterOpt) {
        std::pair<int,int> idAB(kfs.size()-1, kfs.size()-2);
        T_ab = loopClosures[idAB];
      }

      vbo.Upload(kfA.pc_.ptr_,kfA.pc_.SizeBytes(), 0);
      pangolin::glDrawAxis(0.1f);
      glColor4f(1.f,0.f,0.f,0.5f);
      pangolin::RenderVbo(vbo);

      vbo.Upload(kfB.pc_.ptr_,kfB.pc_.SizeBytes(), 0);
      pangolin::glSetFrameOfReference(T_ab.matrix());
      pangolin::glDrawAxis(0.1f);
      glColor4f(0.f,1.f,0.f,0.5f);
      pangolin::RenderVbo(vbo);
      pangolin::glUnsetFrameOfReference();

      glColor4f(1.f,0.f,1.f,0.5f);
      for (size_t i=0; i<assoc_ba.Area(); i+= 300) 
        if (assoc_ba[i] < assoc_ba.Area()) {
          tdp::Vector3fda pb_in_a = T_ab*kfB.pc_[assoc_ba[i]];
          pangolin::glDrawLine(
              kfA.pc_[i](0), kfA.pc_[i](1), kfA.pc_[i](2),
              pb_in_a(0), pb_in_a(1), pb_in_a(2));
        }
    }

    TOCK("Draw 3D");

    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (viewDebugA.IsShown()) {
//      viewDebugA.SetImage(pyrGrey.GetImage(0));
      viewDebugA.SetImage(photoErrBefore);
    }
    if (viewDebugB.IsShown()) {
//      if (kfs.size() > 0)
//        viewDebugB.SetImage(kfs.back().pyrGrey_.GetImage(0));
      viewDebugB.SetImage(photoErrAfter);
    }
    if (viewDebugC.IsShown()) {
      viewDebugC.SetImage(greyA);
    }
    if (viewDebugD.IsShown()) {
      viewDebugD.SetImage(greyB);
    }

    if (viewDepthPyr.IsShown()) {
      tdp::PyramidToImage<float,3>(cuDPyr,dispDepthPyr,
          cudaMemcpyDeviceToHost);
      viewDepthPyr.SetImage(dispDepthPyr);
    }

    if (viewNormalsPyr.IsShown()) {
      if (dispNormalsPyrEst) {
        tdp::PyramidToImage<tdp::Vector3fda,3>(ns_m,cuDispNormalsPyr,
            cudaMemcpyDeviceToDevice);
      } else {
        tdp::PyramidToImage<tdp::Vector3fda,3>(ns_c,cuDispNormalsPyr,
            cudaMemcpyDeviceToDevice);
      }
      tdp::Normals2Image(cuDispNormalsPyr, cuDispNormals2dPyr);
      dispNormals2dPyr.CopyFrom(cuDispNormals2dPyr,cudaMemcpyDeviceToHost);
      viewNormalsPyr.SetImage(dispNormals2dPyr);
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

    if (!gui.paused()) {
      T_mo_prev = T_mo;
    }

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }
  runSave.Set(false);
  workThread.join();
  runLoopClosure.Set(false);
  loopClosureThread.join();
  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;
  return 0;
}

