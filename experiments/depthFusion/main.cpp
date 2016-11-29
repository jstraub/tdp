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
  pangolin::View& viewN3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewN3D);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);
  tdp::ManagedDeviceImage<float> cuGrey(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreydv(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreydu(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuGrad3D(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRGBraw(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRGB(w, h);

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
  tdp::ManagedDeviceImage<float> cuICPassoc_m(wc, hc);
  tdp::ManagedDeviceImage<float> cuICPassoc_c(wc, hc);
  tdp::ManagedHostImage<float> ICPassoc_m(wc, hc);
  tdp::ManagedHostImage<float> ICPassoc_c(wc, hc);

  tdp::ManagedDeviceImage<float> cuPcErr(wc, hc);
  tdp::ManagedDeviceImage<float> cuAngErr(wc, hc);
  tdp::ManagedHostImage<float> pcErr(wc, hc);
  tdp::ManagedHostImage<float> angErr(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3fda> nEstdummy(wc,hc);

  pangolin::GlBufferCudaPtr cuPcbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

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

  tdp::QuickView viewICPassocM(wc,hc);
  gui.container().AddDisplay(viewICPassocM);
  tdp::QuickView viewICPassocC(wc,hc);
  gui.container().AddDisplay(viewICPassocC);

  tdp::QuickView viewAngErr(wc,hc);
  gui.container().AddDisplay(viewAngErr);
  tdp::QuickView viewPcErr(wc,hc);
  gui.container().AddDisplay(viewPcErr);

  tdp::QuickView viewTsdfSliveView(wTSDF,hTSDF);
  gui.container().AddDisplay(viewTsdfSliveView);

  viewICPassocC.Show(true);
  viewICPassocM.Show(true);
  viewAngErr.Show(false);
  viewPcErr.Show(false);

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> tsdfDmin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> tsdfDmax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> dispNormalsPyrEst("ui.disp normal est", false, true);
//  pangolin::Var<bool> dispDepthPyrEst("ui.disp d pyr est", false,true);

  pangolin::Var<bool> runFusion("ui.run Fusion",true,true);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool>  saveTSDF("ui.save TSDF", false, false);
  pangolin::Var<bool> fuseTSDF("ui.fuse TSDF",true,true);
  pangolin::Var<bool> fuseRgb("ui.fuse RGB",false,true);
  pangolin::Var<bool> normalsFromTSDF("ui.normals from TSDF",true,true);
  pangolin::Var<bool> pcFromTSDF("ui.pc from TSDF", true, true);
  pangolin::Var<bool> normalsFromDepthPyr("ui.n from depth pyr",false,true);

  pangolin::Var<float> tsdfMu("ui.mu",0.05,0.01,0.5);
  pangolin::Var<float> tsdfWThr("ui.w thr",25.,1.,20.);
  pangolin::Var<float> tsdfWMax("ui.w max",200.,1.,300.);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",dTSDF/2,0,dTSDF-1);
  pangolin::Var<float> grid0x("ui.grid0 x",-3.0,-2.,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-3.0,-2.,0);
  pangolin::Var<float> grid0z("ui.grid0 z",-3.0,-2.,0);
  pangolin::Var<float> gridEx("ui.gridE x", 3.0,2,3);
  pangolin::Var<float> gridEy("ui.gridE y", 3.0,2,3);
  pangolin::Var<float> gridEz("ui.gridE z", 3.0,2,3);

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  pangolin::Var<bool>  icpGrad3D("ui.run ICP Grad3D", false, true);
  pangolin::Var<float> gradNormThr("ui.grad3d norm thr",6.,0.,10.);

  pangolin::Var<bool>  icpRot("ui.run ICP Rot", false, true);
  pangolin::Var<bool>  icpImu("ui.use IMU to warm start ICP", false, true);
  pangolin::Var<bool>  icpRotOverwrites("ui.ICP Rot Overwrites", false, true);
  pangolin::Var<int>   icpRotIter0("ui.ICP rot iter lvl 0",10,0,10);
  pangolin::Var<int>   icpRotIter1("ui.ICP rot iter lvl 1",7,0,10);
  pangolin::Var<int>   icpRotIter2("ui.ICP rot iter lvl 2",5,0,10);

  pangolin::Var<bool> showIcpError("ui.show ICP",true,true);
  pangolin::Var<int>   icpErrorLvl("ui.ICP error vis lvl",0,0,2);

  pangolin::Var<bool> showPcModel("ui.show model",true,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",true,true);
  pangolin::Var<bool> showPcView("ui.show overview",true,true);

  Stopwatch::getInstance().setCustomSignature(1243984912);

  gui.verbose = true;

  if (false) {
    // for the SR300 or F200
    //depthSensorScale = 1e-4;
    //grid0x = -1.;
    //grid0y = -1.;
    //grid0z = 0.;
    //gridEx = 1.;
    //gridEy = 1.;
    //gridEz = 2.;
    //tsdfMu = 0.2;
    grid0z = 1.;
    gridEz = 6.;
    icpDistThr = 1.;
  }

  tdp::SE3<float> T_mo(Eigen::Matrix4f::Identity());
//  T_mo.rotation() = tdp::SO3f::Rz(M_PI/2.f);
  tdp::SE3f T_mo_0 = T_mo;
  tdp::SE3f T_mo_prev = T_mo_0;
  tdp::SE3f T_wr_imu_prev;
  size_t numFused = 0;

  tdp::SE3f T_wG;

  tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
  tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
  tdp::Vector3fda dGrid = gridE - grid0;

  tdp::ThreadedValue<bool> runWorker(true);
  std::thread workThread([&]() {
        while(runWorker.Get()) {
          if (pangolin::Pushed(saveTSDF)) {
            tdp::ManagedHostVolume<tdp::TSDFval> tmpTSDF(wTSDF, hTSDF, dTSDF);
            tmpTSDF.CopyFrom(cuTSDF, cudaMemcpyDeviceToHost);
            std::cout << "start writing TSDF to " << tsdfOutputPath << std::endl;
            tdp::TSDF::SaveTSDF(tmpTSDF, grid0, dGrid, T_wG, tsdfOutputPath);
            std::cout << "done writing TSDF to " << tsdfOutputPath << std::endl;
          }
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      });

  std::vector<tdp::SE3f> T_mos;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (!normalsFromTSDF) pcFromTSDF = false;
    if (runFusion.GuiChanged() && !runFusion) {
      T_mo_0 = tdp::SE3f();
      T_mo = T_mo_0;
      T_mo_prev = T_mo_0;
    }
    grid0 = tdp::Vector3fda (grid0x,grid0y,grid0z);
    gridE = tdp::Vector3fda (gridEx,gridEy,gridEz);
    dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    if (icpGrad3D) {
      runFusion = false; 
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    // ERIC
    gui.NextFrames();

    tdp::Image<uint16_t> dRaw;
    tdp::Image<tdp::Vector3bda> rgbRaw;

    int64_t t_host_us_d = 0;
    int64_t t_host_us_rgb = 0;

    if (!gui.ImageD(dRaw, 0, &t_host_us_d) ||
        !gui.ImageRGB(rgbRaw, 0, &t_host_us_rgb)) {
      continue;
    }

    tdp::SE3f T_wr_imu = T_ir.Inverse() * imuInterp.Ts_wi_[t_host_us_d * 1000] * T_ir;
    std::cout << " depth frame at " << t_host_us_d << " us" << std::endl;

    if (gui.verbose) std::cout << "setup pyramids" << std::endl;
    TICK("Setup Pyramids");
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    ConvertDepthGpu(cuDraw, cuD, depthSensorScale, tsdfDmin, tsdfDmax);
    cuRGBraw.CopyFrom(rgbRaw, cudaMemcpyHostToDevice);

    // construct pyramid  
    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
        cudaMemcpyDeviceToDevice, 0.03);
    tdp::Depth2PCsGpu(cuDPyr,camD,pcs_c);
    // compute normals
    if (normalsFromDepthPyr) {
      tdp::Depth2NormalsViaDerivativePyr(cuDPyr,camD,ns_c);
    } else { 
      tdp::Depth2Normals(cuDPyr,camD,ns_c);
    }

    if (!gui.ImageRGB(rgb,0)) continue;
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGrey);
    TOCK("Setup Pyramids");

//    if (icpImu && imu) 
//      T_mo = (T_wr_imu * T_wr_imu_prev.Inverse()) * T_mo;
    if (runICP && (!runFusion || numFused > 30)) {
      if (gui.verbose) std::cout << "icp" << std::endl;
      TICK("ICP");
      std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
      tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, T_mo,
          tdp::SE3f(), camD, maxIt, icpAngleThr_deg, icpDistThr,
          gui.verbose); 
      TOCK("ICP");

      std::cout << "T_mo update: " << std::endl 
        << T_mo * T_mo_prev.Inverse() << std::endl;
      std::cout << "IMU : " << std::endl 
        << T_wr_imu * T_wr_imu_prev.Inverse() << std::endl;
    }
    std::cout << "T_mo after ICP: " << std::endl 
      << T_mo  << std::endl;
    T_mos.push_back(T_mo);

    if (showIcpError) {
      tdp::Image<tdp::Vector3fda> pc_m = pcs_m.GetImage(icpErrorLvl);
      ICPassoc_c.Reinitialise(pc_m.w_, pc_m.h_);
      ICPassoc_m.Reinitialise(pc_m.w_, pc_m.h_);
      cuICPassoc_c.Reinitialise(pc_m.w_, pc_m.h_);
      cuICPassoc_m.Reinitialise(pc_m.w_, pc_m.h_);
      ICPassoc_c.Fill(-1.);
      ICPassoc_m.Fill(-1.);
      cuICPassoc_c.CopyFrom(ICPassoc_c,cudaMemcpyHostToDevice);
      cuICPassoc_m.CopyFrom(ICPassoc_m,cudaMemcpyHostToDevice);
      tdp::ICPVisualizeAssoc(pc_m, ns_m.GetImage(icpErrorLvl),
        pcs_c.GetImage(icpErrorLvl), ns_c.GetImage(icpErrorLvl), T_mo,
//        pcs_c.GetImage(icpErrorLvl), ns_c.GetImage(icpErrorLvl), dT,
        tdp::ScaleCamera<float>(camD,pow(0.5,icpErrorLvl)), 
        icpAngleThr_deg, icpDistThr, cuICPassoc_m,
        cuICPassoc_c);
    }

    if (runFusion && (fuseTSDF || numFused <= 30)) {
      if (gui.verbose) std::cout << "add to tsdf" << std::endl;
      TICK("Add To TSDF");
      if (fuseRgb) {
        tdp::TSDF::AddToTSDF(cuTSDF, cuD, cuRGBraw, T_mo, camD, grid0,
            dGrid, tsdfMu); 
      } else {
        tdp::TSDF::AddToTSDF(cuTSDF, cuD, T_mo, camD, grid0, dGrid,
          tsdfMu, tsdfWMax); 
      }
      numFused ++;
      TOCK("Add To TSDF");
    }

    if (runFusion) {
      if (gui.verbose) std::cout << "ray trace" << std::endl;
      TICK("Ray Trace TSDF");
      // first one not needed anymore
      tdp::TSDF::RayTraceTSDF(cuTSDF, pcs_m.GetImage(0), 
            ns_m.GetImage(0), T_mo, camD, grid0, dGrid, tsdfMu, tsdfWThr); 
      // get pc in model coordinate system
      tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_m, cudaMemcpyDeviceToDevice);
      TOCK("Ray Trace TSDF");
      if (normalsFromTSDF) {
        tdp::CompleteNormalPyramid<3>(ns_m, cudaMemcpyDeviceToDevice);
      } else {
        if (normalsFromDepthPyr) {
          tdp::Depth2NormalsViaDerivativePyr(cuDPyrEst,camD,ns_m);
        } else {
          tdp::Depth2Normals(cuDPyrEst,camD,ns_m);
        }
      }
    }

    if (pangolin::Pushed(resetTSDF)) {
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      dEst.Fill(0.);
      cuTSDF.CopyFrom(TSDF, cudaMemcpyHostToDevice);
      numFused = 0;
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
    // Render point cloud from viewpoint of origin
    if (showPcView) {
      tdp::SE3f T_mv;
      tdp::TSDF::RayTraceTSDF(cuTSDF, cuDView, nEstdummy, T_mv, camView, grid0,
          dGrid, tsdfMu, tsdfWThr); 
      tdp::Depth2PCGpu(cuDView,camView,cuPcView);
    }

    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      Eigen::AlignedBox3f box(grid0,gridE);
      glColor4f(1,0,0,0.5f);
      pangolin::glDrawAlignedBox(box);

      // imu
      pangolin::glDrawAxis(T_wr_imu.matrix(),0.2f);

      for (auto& T_moi : T_mos) 
        pangolin::glDrawAxis(T_moi.matrix(),0.1f);

      // render global view of the model first
      pangolin::glDrawAxis(0.1f);
      if (showPcView) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          cudaMemcpy(*cuPcbufp, cuPcView.ptr_, cuPcView.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(0,0,1);
        pangolin::RenderVbo(cuPcbuf);
      }
      // render model and observed point cloud
      if (showPcModel) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = pcs_m.GetImage(icpErrorLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(0,1,0);
        pangolin::RenderVbo(cuPcbuf);
      }
      // render current camera second in the propper frame of
      // reference
      pangolin::glSetFrameOfReference(T_mo.matrix());
      if (showPcCurrent) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = pcs_c.GetImage(icpErrorLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(1,0,0);
        pangolin::RenderVbo(cuPcbuf);
      }
      pangolin::glUnsetFrameOfReference();
    }

    if (viewN3D.IsShown()) {
      viewN3D.Activate(s_cam);
      // render global view of the model first
      pangolin::glDrawAxis(0.1f);
      if (showPcView) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          cudaMemcpy(*cuPcbufp, nEstdummy.ptr_, nEstdummy.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(0,0,1);
        pangolin::RenderVbo(cuPcbuf);
      }
      // render model and observed point cloud
      if (showPcModel) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = ns_m.GetImage(icpErrorLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(0,1,0);
        pangolin::RenderVbo(cuPcbuf);
      }
      Eigen::Matrix4f R_mo = T_mo.matrix();
      R_mo.topRightCorner(3,1).fill(0.);
      pangolin::glSetFrameOfReference(R_mo);
      // render current camera second in the propper frame of
      // reference
      pangolin::glDrawAxis(0.1f);
      if (showPcCurrent) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = ns_c.GetImage(icpErrorLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        glColor3f(1,0,0);
        pangolin::RenderVbo(cuPcbuf);
      }
      pangolin::glUnsetFrameOfReference();
    }
    TOCK("Draw 3D");

    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

//    if (viewTsdfDEst.IsShown()) {
//      tsdfDEst.CopyFrom(cuDEst,cudaMemcpyDeviceToHost);
//      viewTsdfDEst.SetImage(tsdfDEst);
//    }

    if (viewTsdfSliveView.IsShown()) {
      tdp::Image<tdp::TSDFval> cuTsdfSlice =
        cuTSDF.GetImage(std::min((int)cuTSDF.d_-1,tsdfSliceD.Get()));
      tdp::ManagedHostImage<tdp::TSDFval> tsdfSliceRaw(cuTsdfSlice.w_, 
          cuTsdfSlice.h_);
      tsdfSliceRaw.CopyFrom(cuTsdfSlice,cudaMemcpyDeviceToHost);
      for (size_t i=0; i<tsdfSliceRaw.Area(); ++i) 
        tsdfSlice[i] = tsdfSliceRaw[i].f;
      viewTsdfSliveView.SetImage(tsdfSlice);
    }

    if (viewDepthPyr.IsShown()) {
//      if (dispDepthPyrEst) {
//        tdp::PyramidToImage<float,3>(cuDPyrEst,dispDepthPyr,
//            cudaMemcpyDeviceToHost);
//      } else {
        tdp::PyramidToImage<float,3>(cuDPyr,dispDepthPyr,
            cudaMemcpyDeviceToHost);
//      }
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

    if (showIcpError) {
      if (viewICPassocM.IsShown()) {
        ICPassoc_m.CopyFrom(cuICPassoc_m,cudaMemcpyDeviceToHost);
        viewICPassocM.SetImage(ICPassoc_m);
      }
      if (viewICPassocC.IsShown()) {
        ICPassoc_c.CopyFrom(cuICPassoc_c,cudaMemcpyDeviceToHost);
        viewICPassocC.SetImage(ICPassoc_c);
      }
    }

    if (viewPcErr.IsShown()) {
      tdp::L2Distance(pcs_m.GetImage(0), pcs_c.GetImage(0), T_mo, cuPcErr);
      pcErr.CopyFrom(cuPcErr, cudaMemcpyDeviceToHost);
      viewPcErr.SetImage(pcErr);
    }
    if (viewAngErr.IsShown()) {
      tdp::AngularDeviation(ns_m.GetImage(0), ns_c.GetImage(0),
          T_mo.rotation(), cuAngErr);
      angErr.CopyFrom(cuAngErr, cudaMemcpyDeviceToHost);
      viewAngErr.SetImage(angErr);
    }
    TOCK("Draw 2D");

    if (!runFusion) {
      tdp::SO3f R_mo = T_mo.rotation();
      for (size_t lvl=0; lvl<3; ++lvl) {
        tdp::Image<tdp::Vector3fda> pc = pcs_c.GetImage(lvl);
        tdp::Image<tdp::Vector3fda> n = ns_c.GetImage(lvl);
        tdp::Image<tdp::Vector3fda> g = gs_c.GetImage(lvl);
        tdp::TransformPc(T_mo, pc);
        tdp::TransformPc(R_mo, n);
        tdp::TransformPc(R_mo, g);
      }
      pcs_m.CopyFrom(pcs_c,cudaMemcpyDeviceToDevice);
      ns_m.CopyFrom(ns_c,cudaMemcpyDeviceToDevice);
      gs_m.CopyFrom(gs_c,cudaMemcpyDeviceToDevice);
    }
    if (!gui.paused()) {
      T_wr_imu_prev = T_wr_imu;
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
  runWorker.Set(false);
  workThread.join();
  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;
  return 0;
}

