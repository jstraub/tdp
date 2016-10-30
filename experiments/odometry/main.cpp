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
#include <tdp/preproc/depth.h>
#include <tdp/preproc/normals.h>
#include <tdp/preproc/pc.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/manifold/SO3.h>
#include <tdp/preproc/grad.h>
#include <tdp/preproc/grey.h>
#include <tdp/gl/shaders.h>
#include <tdp/utils/colorMap.h>
#include <tdp/camera/photometric.h>

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
  viewNormals.Show(false);

  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
  pangolin::DataLog logInliers;
  pangolin::Plotter plotInliers(&logInliers, -100.f,1.f, 0, 130000.f, 
      10.f, 0.1f);
  plotters.AddDisplay(plotInliers);
  pangolin::DataLog logCost;
  pangolin::Plotter plotCost(&logCost, -100.f,1.f, -10.f,1.f, 10.f, 0.1f);
  plotters.AddDisplay(plotCost);
  pangolin::DataLog logRmse;
  pangolin::Plotter plotRmse(&logRmse, -100.f,1.f, 0.f,0.2f, 0.1f, 0.1f);
  plotters.AddDisplay(plotRmse);
  pangolin::DataLog logOverlap;
  pangolin::Plotter plotOverlap(&logOverlap, -100.f,1.f, 0.f,1.f, .1f, 0.1f);
  plotters.AddDisplay(plotOverlap);
  gui.container().AddDisplay(plotters);

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewModel(wc, hc);
  containerTracking.AddDisplay(viewModel);
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
  tdp::QuickView viewDebugD(3*wc/2, hc);
  containerLoopClosure.AddDisplay(viewDebugD);

  tdp::QuickView viewDebugE(3*wc/2, hc);
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
  tdp::ManagedHostImage<tdp::Vector3bda> rgb_m(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);
  tdp::ManagedDeviceImage<float> cuGrey(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDv(wc,hc);
  tdp::ManagedDeviceImage<float> cuGreyDu(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector2fda> cuGradGrey(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuGrad3D(wc,hc);

  tdp::ManagedHostImage<float> grey_m(wc,hc);

  tdp::ManagedHostImage<float> greyDu(wc, hc);
  tdp::ManagedHostImage<float> greyDv(wc, hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedDeviceImage<float> cuIrmse(wc, hc);
  tdp::ManagedHostImage<float> Irmse(wc, hc);

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

  tdp::ManagedDevicePyramid<float,3> cuPyrGrey_c(wc,hc);
  tdp::ManagedDevicePyramid<float,3> cuPyrGrey_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector2fda,3> cuPyrGradGrey_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector2fda,3> cuPyrGradGrey_m(wc,hc);

  tdp::ManagedDeviceImage<tdp::Vector2fda> cuGrad2D(3*wc/2, hc); 
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuGrad2DImg(3*wc/2, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> grad2DImg(3*wc/2, hc);
  tdp::ManagedHostImage<float> greyPyrImg(3*wc/2, hc); 

  pangolin::GlBufferCudaPtr cuPcbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

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

  pangolin::Var<bool> dispNormalsPyrEst("ui.disp normal est", false, true);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool>  icpFrame2Frame("ui.run frame2frame ICP", true, true);
  pangolin::Var<bool>  icpRgb("ui.run ICP RGB", false, true);
  pangolin::Var<bool>  icpGrad3D("ui.run ICP Grad3D", false, true);
  pangolin::Var<bool>  icpRot("ui.run ICP Rot", false, true);

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  pangolin::Var<float> icpRgbLambda("ui.icp rgb lamb",1.,0.,1.);

  pangolin::Var<float> gradNormThr("ui.grad3d norm thr",6.,0.,10.);

  pangolin::Var<bool> showPcModel("ui.show model",true,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",true,true);

  pangolin::Var<float> rmseView("ui.rmse",0.,0.,0.);

  tdp::SE3f T_wc_0;
  tdp::SE3f T_wc = T_wc_0;
  std::vector<tdp::SE3f> T_wcs;
  tdp::SE3f T_mc; // current to model

  gui.verbose = false;
  if (gui.verbose) std::cout << "starting main loop" << std::endl;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {

    if (pangolin::Pushed(resetICP)) {
      T_mc = tdp::SE3f();
      T_wc = T_wc_0;
      T_wcs.clear();
      gui.Seek(0);
    }

    if (gui.frame > 0 && !gui.paused()) { 
//      for (size_t lvl=0; lvl<3; ++lvl) {
//        tdp::Image<tdp::Vector3fda> pc = pcs_c.GetImage(lvl);
//        tdp::Image<tdp::Vector3fda> n = ns_c.GetImage(lvl);
//        tdp::Image<tdp::Vector3fda> g = gs_c.GetImage(lvl);
//        tdp::TransformPc(T_wc, pc);
//        tdp::TransformPc(T_wc.rotation(), n);
//        tdp::TransformPc(T_wc.rotation(), g);
//      }
      pcs_m.CopyFrom(pcs_c,cudaMemcpyDeviceToDevice);
      ns_m.CopyFrom(ns_c,cudaMemcpyDeviceToDevice);
      gs_m.CopyFrom(gs_c,cudaMemcpyDeviceToDevice);
      cuPyrGrey_m.CopyFrom(cuPyrGrey_c, cudaMemcpyDeviceToDevice);
      cuPyrGradGrey_m.CopyFrom(cuPyrGradGrey_c, cudaMemcpyDeviceToDevice);
      rgb_m.CopyFrom(rgb, cudaMemcpyHostToHost);
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();

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
    
    tdp::Image<tdp::Vector2fda> cuGradGrey_c = cuPyrGradGrey_c.GetImage(0);
    tdp::Gradient(cuGrey, cuGreyDu, cuGreyDv, cuGradGrey_c);
    greyDu.CopyFrom(cuGreyDu, cudaMemcpyDeviceToHost);
    greyDv.CopyFrom(cuGreyDv, cudaMemcpyDeviceToHost);
    tdp::ConstructPyramidFromImage(cuGrey, cuPyrGrey_c,
        cudaMemcpyDeviceToDevice);
    tdp::CompletePyramid(cuPyrGradGrey_c, cudaMemcpyDeviceToDevice);

//    tdp::Image<tdp::Vector3fda> cuNs = ns_c.GetImage(0);
//    tdp::Image<tdp::Vector3fda> cuGs = gs_c.GetImage(0);
//    tdp::Gradient3D(cuGrey, cuD, cuNs, camD, gradNormThr, cuGreydu,
//        cuGreydv, cuGs);
//    tdp::CompletePyramid(gs_c, cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

    std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
    std::vector<float> errPerLvl;
    std::vector<float> countPerLvl;
    T_mc = tdp::SE3f();
    Eigen::Matrix<float,6,6> Sigma_mc = 1e-4*Eigen::Matrix<float,6,6>::Identity();
    TICK("ICP");
    if(icpFrame2Frame) {
      if (gui.verbose) std::cout << "icp" << std::endl;
      tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_c, ns_c,
          rig, rig.rgbStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
          gui.verbose, T_mc, Sigma_mc, errPerLvl, countPerLvl);
    } else if (icpRgb) {
      tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m,
          cuPyrGradGrey_m, cuPyrGrey_m, pcs_c, ns_c, cuPyrGradGrey_c,
          cuPyrGrey_c, rig, rig.rgbStream2cam_, maxIt, icpAngleThr_deg,
          icpDistThr, icpRgbLambda, gui.verbose, T_mc, Sigma_mc,
          errPerLvl, countPerLvl);
    } else if (icpGrad3D) {
      tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_c, ns_c,
          rig, rig.dStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
          gui.verbose, T_mc, Sigma_mc, errPerLvl, countPerLvl);
    } else if (icpRot) {
      tdp::ICP::ComputeProjectiveRotation<CameraT::NumParams,CameraT>(
          ns_m,  ns_c, pcs_c, 
          T_mc, rig.T_rcs_[rig.rgbStream2cam_[0]].Inverse(),
          rig.cams_[rig.rgbStream2cam_[0]],
          maxIt, icpAngleThr_deg);
    }
    TOCK("ICP");
    if (!gui.paused()) {
      T_wc = T_wc * T_mc;
      T_wcs.push_back(T_wc);
    }

    logInliers.Log(countPerLvl);
    logCost.Log(errPerLvl);

    float overlap, rmse;
    cudaMemset(cuIrmse.ptr_, 0, cuIrmse.SizeBytes());
    tdp::OverlapGpu(cuPyrGrey_m.GetImage(0), cuPyrGrey_c.GetImage(0),
        pcs_m.GetImage(0), pcs_c.GetImage(0), T_mc, rig,
        overlap, rmse, &cuIrmse); 
//    tdp::OverlapGpu(cuPyrGrey_m.GetImage(0), cuPyrGrey_c.GetImage(0),
//        pcs_m.GetImage(0), pcs_c.GetImage(0),
//        rig.T_rcs_[rig.rgbStream2cam_[0]].Inverse()* T_mc, 
//        rig.cams_[rig.rgbStream2cam_[0]],
//        overlap, rmse, &cuIrmse); 
    Irmse.CopyFrom(cuIrmse, cudaMemcpyDeviceToHost);
    rmseView = rmse;
    logRmse.Log(rmse);
    logOverlap.Log(overlap);

    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    TICK("Draw 3D");
    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      pangolin::glDrawAxis(T_wc.matrix(), 0.05f);
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,30);

      // render model and observed point cloud
      if (showPcModel) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = pcs_m.GetImage(dispLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        cbo.Upload(rgb_m.ptr_, rgb_m.SizeBytes(), 0);
        glColor3f(0,1,0);
        pangolin::glSetFrameOfReference((T_wc*T_mc.Inverse()).matrix());
        pangolin::RenderVboCbo(cuPcbuf, cbo, true);
        pangolin::glUnsetFrameOfReference();
      }
      // render current camera second in the propper frame of
      // reference
      if (showPcCurrent) {
        {
          pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
          cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
          tdp::Image<tdp::Vector3fda> pc0 = pcs_c.GetImage(dispLvl);
          cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
              cudaMemcpyDeviceToDevice);
        }
        cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
        glColor3f(1,0,0);
        pangolin::glSetFrameOfReference(T_wc.matrix());
        pangolin::RenderVboCbo(cuPcbuf, cbo, true);
        pangolin::glUnsetFrameOfReference();
      }
    }

    TOCK("Draw 3D");

    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (viewModel.IsShown()) {
      grey_m.CopyFrom(cuPyrGrey_m.GetImage(0), cudaMemcpyDeviceToHost);
      viewModel.SetImage(grey_m);
    }
    if (viewCurrent.IsShown()) {
      viewCurrent.SetImage(rgb);
    }
    if (viewDebugA.IsShown()) {
      viewDebugA.SetImage(greyDu);
    }
    if (viewDebugB.IsShown()) {
      viewDebugB.SetImage(greyDv);
    }
    if (viewDebugC.IsShown()) {
      viewDebugC.SetImage(Irmse);
    }
    if (viewDebugD.IsShown()) {
      tdp::PyramidToImage(cuPyrGradGrey_c, cuGrad2D, cudaMemcpyDeviceToDevice);
      tdp::Grad2Image(cuGrad2D, cuGrad2DImg);
      grad2DImg.CopyFrom(cuGrad2DImg, cudaMemcpyDeviceToHost);
      viewDebugD.SetImage(grad2DImg);
    }
    if (viewDebugE.IsShown()) {
      tdp::PyramidToImage(cuPyrGrey_c, greyPyrImg, cudaMemcpyDeviceToHost);
      viewDebugE.SetImage(greyPyrImg);
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

    plotInliers.ScrollView(1,0);
    plotCost.ScrollView(1,0);
    plotRmse.ScrollView(1,0);
    plotOverlap.ScrollView(1,0);

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

  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;
  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

