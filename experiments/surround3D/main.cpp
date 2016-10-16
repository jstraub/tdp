/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <vector>
#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image_io.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/data/managed_volume.h>

#include <tdp/tsdf/tsdf.h>
#include <tdp/icp/icp.h>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#include <tdp/utils/Stopwatch.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/gui/gui_base.hpp>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SE3.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/geometry/cosy.h>

#include <tdp/io/tinyply.h>
#include <tdp/slam/keyframe.h>
#include <tdp/slam/keyframe_slam.h>

typedef tdp::CameraPoly3<float> CameraT;
//typedef tdp::Camera<float> CameraT;
//
int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  std::string input_uri = std::string(argv[1]);
  std::string configPath = std::string(argv[2]);
  std::string imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  std::string output_uri = (argc > 4) ? std::string(argv[4]) : dflt_output_uri;
  std::string tsdfOutputPath = "tsdf.raw";

  bool keepRunningWhilePaused = false;

  pangolin::Uri uri = pangolin::ParseUri(input_uri);
  if (!uri.scheme.compare("file")) {
    std::cout << uri.scheme << std::endl; 
    if (pangolin::FileExists(uri.url+std::string("imu.pango"))
     && pangolin::FileExists(uri.url+std::string("video.pango"))) {
      imu_input_uri = input_uri + std::string("imu.pango");
      tsdfOutputPath = uri.url + tsdfOutputPath;
      input_uri = input_uri + std::string("video.pango");
    } else if (pangolin::FileExists(uri.url+std::string("video.pango"))) {
      input_uri = input_uri + std::string("video.pango");
    } 
  }

  std::cout << input_uri << std::endl;
  std::cout << imu_input_uri << std::endl;

  std::cout << " -!!- this application works only with openni2 devices (tested with Xtion PROs) -!!- " << std::endl;

  // Read rig file
  tdp::Rig<CameraT> rig;
  if (!rig.FromFile(configPath, false)) {
    pango_print_error("No config file specified.\n");
    return 1;
  }

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 2;
  }

  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  rig.CorrespondOpenniStreams2Cams(streams);


  // optionally connect to IMU if it is found.
  tdp::ImuInterface* imu = nullptr; 
  if (imu_input_uri.size() > 0) 
    imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();
  tdp::ImuInterpolator imuInterp(imu,nullptr);
  imuInterp.Start();
  
  tdp::SE3f T_ir;
  if (imu) {
    if (rig.T_ris_.size() > 0) 
      T_ir = rig.T_ris_[0];
    else {
      std::cout << "Warning no IMU calibration specified" << std::endl;
    }
  }

  tdp::GuiBase gui(1200,800,video);
  Stopwatch::getInstance().setCustomSignature(1237249810);

  size_t wSingle = video.Streams()[0].Width();
  size_t hSingle = video.Streams()[0].Height();
  wSingle += wSingle%64;
  hSingle += hSingle%64;
  size_t w = wSingle;
  size_t h = 3*hSingle;
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  w += w%64;
  h += h%64;
  size_t dTSDF = 512;
  size_t wTSDF = 512;
  size_t hTSDF = 512;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,3*480,420,3*420,320,3*240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewMain3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewMain3D);

  pangolin::OpenGlRenderState camLoopClose(
      pangolin::ProjectionMatrix(640,3*480,420,3*420,320,3*240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  pangolin::View& viewLoopClose = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(camLoopClose));
  gui.container().AddDisplay(viewLoopClose);

  tdp::QuickView viewRgb(w,h);
  gui.container().AddDisplay(viewRgb);
  tdp::QuickView viewD(w,h);
  gui.container().AddDisplay(viewD);
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);

  tdp::QuickView viewDebug(640,480);
  gui.container().AddDisplay(viewDebug);

  viewRgb.Show(false);
  viewD.Show(false);
  viewN2D.Show(false);
  viewDebug.Show(false);

  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
  pangolin::DataLog logInliers;
  pangolin::Plotter plotInliers(&logInliers, -100.f,1.f, 0, 130000.f, 
      10.f, 0.1f);
  plotters.AddDisplay(plotInliers);
  pangolin::DataLog logCost;
  pangolin::Plotter plotCost(&logCost, -100.f,1.f, -10.f,1.f, 10.f, 0.1f);
  plotters.AddDisplay(plotCost);
  gui.container().AddDisplay(plotters);

  viewRgb.Show(false);
  viewD.Show(false);
  viewN2D.Show(false);

  tdp::Camera<float> camView(Eigen::Vector4f(220,220,319.5,239.5)); 
  tdp::ManagedDeviceImage<float> cuDView(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcView(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> nEstdummy(w,h);
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> n(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

  // loop closure pc
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcA(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNA(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcB(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNB(w, h);

  tdp::ManagedHostImage<int> assoc_ba(w,h);
  tdp::ManagedDeviceImage<int> cuAssoc_ba(w,h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPc(w, h);
  tdp::ManagedDeviceImage<float> cuScale(w,h);

  tdp::ManagedHostPyramid<float,3> dPyr(w,h);
  tdp::ManagedHostPyramid<float,3> dPyrEst(w,h);
  tdp::ManagedDevicePyramid<float,3> cuDPyr(w,h);
  tdp::ManagedDevicePyramid<float,3> cuDPyrEst(w,h);
  // model pc
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(w,h);
  // current pc
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_o(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_o(w,h);
  // loop closure pc
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_l(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_l(w,h);

  tdp::ManagedHostVolume<tdp::TSDFval> TSDF(wTSDF, hTSDF, dTSDF);
  TSDF.Fill(tdp::TSDFval(-1.01,0.));
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTSDF(wTSDF, hTSDF, dTSDF);
  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,10.);

  pangolin::Var<bool> useRgbCamParasForDepth("ui.use rgb cams", true, true);

  pangolin::Var<bool> odomImu("ui.odom IMU", false, true);
  pangolin::Var<bool> odomFrame2Frame("ui.odom frame2frame", true, true);
  pangolin::Var<bool> odomFrame2Model("ui.odom frame2model", false, true);
  pangolin::Var<bool> resetOdom("ui.reset odom",false,false);

  pangolin::Var<bool> savePC("ui.save current PC",false,false);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool>  saveTSDF("ui.save TSDF", false, false);
  pangolin::Var<bool> fuseTSDF("ui.fuse TSDF",true,true);
  pangolin::Var<float> tsdfMu("ui.mu",0.5,0.,1.);
  pangolin::Var<float> grid0x("ui.grid0 x",-5.0,-2,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-5.0,-2,0);
  pangolin::Var<float> grid0z("ui.grid0 z",-5.0,-2,0);
  pangolin::Var<float> gridEx("ui.gridE x",5.0,2,0);
  pangolin::Var<float> gridEy("ui.gridE y",5.0,2,0);
  pangolin::Var<float> gridEz("ui.gridE z",5.0,2,0);

  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<bool>  icpImu("ui.use IMU to warm start ICP", false, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  pangolin::Var<int>   inlierThrLvl0("ui.inlier thr lvl 0", 10000, 1000, 100000);

  pangolin::Var<bool> dispEst("ui.disp Est", false,true);

  pangolin::Var<bool> keyFrameSLAM("ui.keyframe SLAM", true, true);
  pangolin::Var<bool> tryLoopClose("ui.loop close", false,true);
  pangolin::Var<float>  keyFrameDistThresh("ui.KF dist thr", 0.3, 0.01, 0.5);
  pangolin::Var<float>  keyFrameAngleThresh("ui.KF angle thr", 20, 1, 50);
  pangolin::Var<int>  icpDownSample("ui.ICP downsample",30,1,100);
  pangolin::Var<float> icpLoopCloseAngleThr_deg("ui.icpLoop angle thr",30,0.,90.);
  pangolin::Var<float> icpLoopCloseDistThr("ui.icpLoop dist thr",0.50,0.,1.);
  pangolin::Var<int>   icpLoopCloseIter0("ui.icpLoop iter lvl 0",8,0,10);

  pangolin::RegisterKeyPressCallback('c', [&](){
      for (size_t sId=0; sId < rig.rgbdStream2cam_.size(); sId++) {
        int cId = rig.rgbdStream2cam_[sId];
        std::stringstream ss;
        ss << "capture_cam" << cId << ".png";
        try{
          pangolin::SaveImage(
            gui.images[gui.iRGB[sId]], 
            gui.video.Streams()[gui.iRGB[sId]].PixFormat(),
            pangolin::MakeUniqueFilename(ss.str())
            );
        }catch(std::exception e){
          pango_print_error("Unable to save frame: %s\n", e.what());
        }
      }
    });

  pangolin::GlRenderBuffer glRenderBuf(w,h);
  pangolin::GlTexture tex(w,h,GL_RGBA8);
  pangolin::GlFramebuffer glFrameBuf(tex, glRenderBuf);
  tdp::ManagedHostImage<tdp::Vector3bda> rgbJoint(w,h);
  memset(rgbJoint.ptr_, 0, rgbJoint.SizeBytes());
  tdp::ManagedHostImage<float> dJoint(w,h);

  tdp::ThreadedValue<bool> runWorker(true);
  std::thread workThread([&]() {
        while(runWorker.Get()) {
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

  tdp::SE3f T_mr0;
  if (imu) {
    tdp::SO3f R_im (tdp::OrthonormalizeFromYZ(Eigen::Vector3f(0,1,0), 
        -imuInterp.gravity0_));
    tdp::SE3f T_im(R_im);
    T_mr0 = T_im.Inverse() * T_ir;
    std::cout << "found IMU and used gravity estimate " 
      << imuInterp.gravity0_.transpose() << std::endl
      << T_mr0 << std::endl;
//    T_mr.matrix().topLeftCorner(3,3) =
//      tdp::Orthonormalize(Eigen::Vector3f(1,0,0), imuInterp.gravity0_);
//    std::cout << "found IMU and used gravity estimate " 
//      << imuInterp.gravity0_.transpose() << std::endl
//      << T_mr << std::endl;
  }
  tdp::KeyframeSLAM kfSLAM;
  std::map<std::pair<int,int>,tdp::SE3f> loopClosures;
  std::vector<tdp::KeyFrame> keyframes;
  tdp::SE3f T_mr = T_mr0;
  std::vector<tdp::SE3f> T_mrs;
  std::vector<tdp::SE3f> T_wr_imus;
  tdp::SE3f T_wr_imu_prev;
  size_t numFused = 0;
  gui.verbose = false;
  // Stream and display video
  while(!pangolin::ShouldQuit() && (keepRunningWhilePaused || !gui.finished()))
  {
    tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
    tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
    tdp::Vector3fda dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    if (odomFrame2Model.GuiChanged() && odomFrame2Model) {
      odomImu = false;
      odomFrame2Frame = false;
    } else if (odomFrame2Frame.GuiChanged() && odomFrame2Frame) {
      odomImu = false;
      odomFrame2Model = false;
    } else if (odomImu.GuiChanged() && odomImu) {
      odomFrame2Frame = false;
      odomFrame2Model = false;
    }

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    TICK("rgb collection");
    rig.CollectRGB(gui, rgb, cudaMemcpyHostToHost);
    TOCK("rgb collection");
    TICK("depth collection");
    int64_t t_host_us_d = 0;
    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
    TOCK("depth collection");
    TICK("pc and normals");
    rig.ComputePc(cuD, useRgbCamParasForDepth, cuPc);
    rig.ComputeNormals(cuD, useRgbCamParasForDepth, cuN);
    TOCK("pc and normals");
//    if (gui.verbose) std::cout << "ray trace tsdf" << std::endl;
    TICK("Setup Pyramids");
    // TODO might want to use the pyramid construction with smoothing
    pcs_o.GetImage(0).CopyFrom(cuPc, cudaMemcpyDeviceToDevice);
    tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_o,cudaMemcpyDeviceToDevice);
    ns_o.GetImage(0).CopyFrom(cuN, cudaMemcpyDeviceToDevice);
    tdp::CompleteNormalPyramid<3>(ns_o,cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

    tdp::SE3f T_wr_imu = T_mr0*T_ir.Inverse()*imuInterp.Ts_wi_[t_host_us_d*1000]*T_ir;
    if (odomImu) {
//      T_mr = (T_wr_imu * T_wr_imu_prev.Inverse()) * T_mr;
      T_mr = T_wr_imu;
    } else if (odomFrame2Model || odomFrame2Frame) {
      if (runICP && numFused > 30 && !gui.paused()) {
//        if (gui.verbose) std::cout << "icp" << std::endl;
        if (icpImu && imu) 
          T_mr = (T_wr_imu * T_wr_imu_prev.Inverse()) * T_mr;
        TICK("ICP");
        std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
        std::vector<float> errPerLvl;
        std::vector<float> countPerLvl;
        if (useRgbCamParasForDepth) {
          tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_o, ns_o,
              rig, rig.rgbStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
              gui.verbose, T_mr, errPerLvl, countPerLvl);
        } else {
          tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_o, ns_o,
              rig, rig.dStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
              gui.verbose, T_mr, errPerLvl, countPerLvl);
        }
        logInliers.Log(countPerLvl);
        logCost.Log(errPerLvl);
        if (countPerLvl[0] < inlierThrLvl0 
            || errPerLvl[0] != errPerLvl[0]
            || errPerLvl[1] != errPerLvl[1]
            || errPerLvl[2] != errPerLvl[2]) {
          std::cout << "# inliers " << countPerLvl[0] << " to small "
            << "probably have tracking failure"
            << std::endl;
          gui.pause();
          runICP = false;
          fuseTSDF = false;
        } 
        TOCK("ICP");
      }
    }
    T_mrs.push_back(T_mr);
    // Get translation from T_mr
    T_wr_imu.matrix().topRightCorner(3,1) = T_mr.translation();
    T_wr_imus.push_back(T_wr_imu);

    if (pangolin::Pushed(resetTSDF)) {
      T_mr.matrix() = Eigen::Matrix4f::Identity();
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
      numFused = 0;
    }
    if (pangolin::Pushed(resetOdom)) {
      T_mr.matrix() = Eigen::Matrix4f::Identity();
    }

    if (!gui.paused() && (fuseTSDF || numFused <= 30)) {
//      if (gui.verbose) std::cout << "add to tsdf" << std::endl;
      TICK("Add To TSDF");
      for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
        int32_t cId;
        if (useRgbCamParasForDepth) {
          cId = rig.rgbStream2cam_[sId]; 
        } else {
          cId = rig.dStream2cam_[sId]; 
        }
        CameraT cam = rig.cams_[cId];
        tdp::SE3f T_rc = rig.T_rcs_[cId];
        tdp::SE3f T_mo = T_mr+T_rc;
        tdp::Image<float> cuD_i(wSingle, hSingle,
            cuD.ptr_+rig.rgbdStream2cam_[sId]*wSingle*hSingle);
        AddToTSDF(cuTSDF, cuD_i, T_mo, cam, grid0, dGrid, tsdfMu); 
      }
      numFused ++;
      TOCK("Add To TSDF");
    }

    if (!gui.paused() && (odomImu || odomFrame2Model)) {
      TICK("Ray Trace TSDF");
      tdp::Image<tdp::Vector3fda> cuNEst = ns_m.GetImage(0);
      tdp::Image<tdp::Vector3fda> cuPcEst = pcs_m.GetImage(0);
      for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
        int32_t cId;
        if (useRgbCamParasForDepth) {
          cId = rig.rgbStream2cam_[sId]; 
        } else {
          cId = rig.dStream2cam_[sId]; 
        }
        CameraT cam = rig.cams_[cId];
        tdp::SE3f T_rc = rig.T_rcs_[cId];
        tdp::SE3f T_mo = T_mr*T_rc;

        tdp::Image<tdp::Vector3fda> cuNEst_i = cuNEst.GetRoi(0,
            rig.rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
        tdp::Image<tdp::Vector3fda> cuPcEst_i = cuPcEst.GetRoi(0,
            rig.rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);

        // ray trace the TSDF to get pc and normals in model cosy
        RayTraceTSDF(cuTSDF, cuPcEst_i, 
            cuNEst_i, T_mo, cam, grid0, dGrid, tsdfMu); 
      }
      // just complete the surface normals obtained from the TSDF
      tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_m,cudaMemcpyDeviceToDevice);
      tdp::CompleteNormalPyramid<3>(ns_m,cudaMemcpyDeviceToDevice);
      TOCK("Ray Trace TSDF");
    }

    if (keyFrameSLAM) {
      Eigen::Matrix<float,6,1> se3 = Eigen::Matrix<float,6,1>::Zero();
      if (keyframes.size() > 0)
        se3 = keyframes.back().T_wk_.Log(T_mr);
      if ((keyframes.size() == 0 && gui.frame > 10)
          || se3.head<3>().norm()*180./M_PI > keyFrameAngleThresh
          || se3.tail<3>().norm() >  keyFrameDistThresh) {
        std::cout << "adding keyframe " << keyframes.size() 
          << " angle: " << se3.head<3>().norm()*180./M_PI 
          << " dist: " << se3.tail<3>().norm() << std::endl;
        pc.CopyFrom(pcs_o.GetImage(0),cudaMemcpyDeviceToHost);
        n.CopyFrom(ns_o.GetImage(0),cudaMemcpyDeviceToHost);
        kfSLAM.AddKeyframe(pc, n, rgb, T_mr);
        keyframes.emplace_back(pc, n, rgb, T_mr);
        tryLoopClose = true;
      }
    }

    if (pangolin::Pushed(tryLoopClose) && keyframes.size() > 1) {
      int idA = keyframes.size()-1;
      tdp::KeyFrame& kfA = keyframes[idA];
      cuPcA.CopyFrom(kfA.pc_, cudaMemcpyHostToDevice);
      cuNA.CopyFrom(kfA.n_, cudaMemcpyHostToDevice);
      for (int idB=(int)keyframes.size()-2; 
          idB>std::max(-1,(int)keyframes.size()-1-10); --idB) {
        tdp::KeyFrame& kfB = keyframes[idB];
        cuPcB.CopyFrom(kfB.pc_, cudaMemcpyHostToDevice);
        cuNB.CopyFrom(kfB.n_, cudaMemcpyHostToDevice);

        tdp::SE3f T_ab = kfA.T_wk_.Inverse() * kfB.T_wk_;
        std::cout << keyframes.size()-2 << " to " 
          << keyframes.size()-1 
          << " Initial transformation: " << std::endl 
          << T_ab.matrix3x4() << std::endl;
        float err;
        float count;
        tdp::ICP::ComputeANN(kfA.pc_, cuPcA, cuNA, kfB.pc_, cuPcB, cuNB, 
            assoc_ba, cuAssoc_ba, T_ab, icpLoopCloseIter0, 
            icpLoopCloseAngleThr_deg, icpLoopCloseDistThr, 
            icpDownSample, gui.verbose, err, count);
        if (err == err && count > 3000) {
          std::cout << "successfull loop closure "
            << T_ab.matrix3x4() << std::endl;
          loopClosures.emplace(std::make_pair(idA, idB), T_ab);
          kfSLAM.AddLoopClosure(idA, idB, T_ab);
        }
      }
      kfSLAM.Optimize(); 
    }

    // Render point cloud from viewpoint of origin
    tdp::SE3f T_mv;
    T_mv.matrix()(2,3) = -3.;
    RayTraceTSDF(cuTSDF, cuDView, nEstdummy, T_mv, camView, grid0,
        dGrid, tsdfMu); 
    tdp::Depth2PCGpu(cuDView,camView,cuPcView);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    if (viewMain3D.IsShown()) {
      if (dispEst) {
        pc.CopyFrom(pcs_m.GetImage(0),cudaMemcpyDeviceToHost);
      } else {
        pc.CopyFrom(pcs_o.GetImage(0),cudaMemcpyDeviceToHost);
      }
      viewMain3D.Activate(s_cam);
      // draw the axis
      pangolin::glDrawAxis(0.1);
      if (!keyFrameSLAM) {
        for (size_t i=0; i<T_mrs.size(); ++i) {
          if (i%10==0) 
            pangolin::glDrawAxis(T_mrs[i].matrix(), 0.1f);
          glColor4f(1.,1.,0.,0.6);
          if (i>0) {
            pangolin::glDrawLine(
                T_mrs[i].translation()(0),
                T_mrs[i].translation()(1),
                T_mrs[i].translation()(2),
                T_mrs[i-1].translation()(0),
                T_mrs[i-1].translation()(1),
                T_mrs[i-1].translation()(2));
          }
        }
        for (size_t i=0; i<T_wr_imus.size(); ++i) {
          if (i%10==0) 
            pangolin::glDrawAxis(T_wr_imus[i].matrix(), 0.1f);
          glColor4f(0.,1.,1.,0.6);
        }
      } else {
        for (size_t i=0; i<T_mrs.size(); ++i) {
          glColor4f(1.,1.,0.,0.6);
          if (i>0) {
            pangolin::glDrawLine(
                T_mrs[i].translation()(0), T_mrs[i].translation()(1),
                T_mrs[i].translation()(2),
                T_mrs[i-1].translation()(0), T_mrs[i-1].translation()(1),
                T_mrs[i-1].translation()(2));
          }
        }
        for (size_t i=0; i<keyframes.size(); ++i) {
          tdp::SE3f& T_wk = keyframes[i].T_wk_;
          pangolin::glDrawAxis(T_wk.matrix(), 0.1f);
        }
        glColor4f(1.,0.3,0.3,0.6);
        for (auto& it : loopClosures) {
          tdp::SE3f& T_wk_A = keyframes[it.first.first].T_wk_;
          tdp::SE3f& T_wk_B = keyframes[it.first.second].T_wk_;
          pangolin::glDrawLine(
              T_wk_A.translation()(0), T_wk_A.translation()(1),
              T_wk_A.translation()(2),
              T_wk_B.translation()(0), T_wk_B.translation()(1),
              T_wk_B.translation()(2));
        }
      }

      Eigen::AlignedBox3f box(grid0,gridE);
      glColor4f(1,0,0,0.5f);
      pangolin::glDrawAlignedBox(box);

      vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
      // render point cloud
      if (dispEst) {
        pangolin::RenderVboCbo(vbo,cbo,true);
      }
      pangolin::glSetFrameOfReference(T_mr.matrix());
      pangolin::glDrawAxis(0.1f);
      if (!dispEst) {
        pangolin::RenderVboCbo(vbo,cbo,true);
      }
      pangolin::glUnsetFrameOfReference();

      pc.CopyFrom(cuPcView,cudaMemcpyDeviceToHost);
      vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
      pangolin::glDrawAxis(0.1f);
      glColor4f(1.f,0.f,0.f,0.5f);
      pangolin::glSetFrameOfReference(T_mv.matrix());
      pangolin::RenderVbo(vbo);
      pangolin::glUnsetFrameOfReference();
    }

    if (viewLoopClose.IsShown() && keyframes.size() > 1) {
      viewLoopClose.Activate(camLoopClose);
      tdp::KeyFrame& kfA = keyframes[keyframes.size()-1];
      tdp::KeyFrame& kfB = keyframes[keyframes.size()-2];
      tdp::SE3f T_ab = kfA.T_wk_.Inverse() * kfB.T_wk_;

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
    glDisable(GL_DEPTH_TEST);

    // Draw 2D stuff
    if (viewRgb.IsShown()) {
      viewRgb.SetImage(rgb);
    }
    if (viewD.IsShown()) {
      if (dispEst) {
        d.CopyFrom(cuDPyrEst.GetImage(0), cudaMemcpyDeviceToHost);
      }else {
        d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
      }
      viewD.SetImage(d);
    }
    if (viewN2D.IsShown()) {
      // convert normals to RGB image
      if (dispEst) {
        tdp::Normals2Image(ns_m.GetImage(0), cuN2D);
      } else {
        tdp::Normals2Image(cuN, cuN2D);
      }
      n2D.CopyFrom(cuN2D,cudaMemcpyDeviceToHost);
      viewN2D.SetImage(n2D);
    }
    if (viewDebug.IsShown()) {
      tdp::ManagedHostImage<float> debug(640,480);
      debug.CopyFrom(rig.cuDepthScales_[0], cudaMemcpyDeviceToHost);
      viewDebug.SetImage(debug);
    }

    plotInliers.ScrollView(1,0);
    plotCost.ScrollView(1,0);

    if (pangolin::Pushed(savePC)) {

      pc.CopyFrom(pcs_o.GetImage(0),cudaMemcpyDeviceToHost);
      n.CopyFrom(ns_o.GetImage(0),cudaMemcpyDeviceToHost);

      std::vector<std::string> comments;
      std::stringstream ss;
      ss << "frame" << gui.frame;
      comments.push_back("generated from surround3D");
      comments.push_back("PC in rig coordinate system");
      comments.push_back(input_uri);
      comments.push_back(configPath);
      comments.push_back(ss.str());

      SavePointCloud(uri.url+ss.str()+std::string(".ply"),
          pc, n, comments);
    }

    if (odomFrame2Frame) {
      for (size_t lvl=0; lvl<3; ++lvl) {
        tdp::Image<tdp::Vector3fda> pc = pcs_o.GetImage(lvl);
        tdp::Image<tdp::Vector3fda> n = ns_o.GetImage(lvl);
        tdp::TransformPc(T_mr, pc);
        tdp::TransformPc(T_mr.rotation(), n);
      }
      pcs_m.CopyFrom(pcs_o, cudaMemcpyDeviceToDevice);
      ns_m.CopyFrom(ns_o, cudaMemcpyDeviceToDevice);
    }
    if (!gui.paused()) {
      T_wr_imu_prev = T_wr_imu;
    }

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // if we are recording
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    // finish this frame
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }
  saveTSDF = true;

  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;

  std::this_thread::sleep_for(std::chrono::microseconds(500));
  runWorker.Set(false);
  workThread.join();
  return 0;
}

