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

#include "gui.hpp"
#include <tdp/camera/rig.h>
#include <tdp/marker/aruco.h>
#include <tdp/manifold/SE3.h>
#include <pangolin/video/drivers/openni2.h>

typedef tdp::CameraPoly3<float> CameraT;
//typedef tdp::Camera<float> CameraT;

void VideoViewer(const std::string& input_uri, 
    const std::string& configPath,
    const std::string& output_uri)
{
  std::cout << " -!!- this application works only with openni2 devices (tested with Xtion PROs) -!!- " << std::endl;

  // Read rig file
  tdp::Rig<CameraT> rig;
  if (!rig.FromFile(configPath, false)) return;

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return;
  }
  std::vector<int32_t> rgbStream2cam;
  std::vector<int32_t> dStream2cam;
  std::vector<int32_t> rgbdStream2cam;
  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  tdp::CorrespondOpenniStreams2Cams(streams,rig,rgbStream2cam,
      dStream2cam, rgbdStream2cam);

  tdp::GUInoViews gui(1200,800,video);

  size_t wSingle = video.Streams()[0].Width();
  size_t hSingle = video.Streams()[0].Height();
  size_t w = wSingle;
  size_t h = 3*hSingle;
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  w += w%64;
  h += h%64;
  size_t dTSDF = 128;
  size_t wTSDF = 512;
  size_t hTSDF = 512;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,3*480,420,3*420,320,3*240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);

  tdp::QuickView viewRgb(w,h);
  gui.container().AddDisplay(viewRgb);
  tdp::QuickView viewD(w,h);
  gui.container().AddDisplay(viewD);
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);

  tdp::QuickView viewRgbJoint(w,h);
  gui.container().AddDisplay(viewRgbJoint);
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

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
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_o(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_o(w,h);

  tdp::ManagedHostVolume<float> W(wTSDF, hTSDF, dTSDF);
  tdp::ManagedHostVolume<float> TSDF(wTSDF, hTSDF, dTSDF);
  W.Fill(0.);
  TSDF.Fill(-1.01);
  tdp::ManagedDeviceVolume<float> cuW(wTSDF, hTSDF, dTSDF);
  tdp::ManagedDeviceVolume<float> cuTSDF(wTSDF, hTSDF, dTSDF);

  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
  tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  //pangolin::Var<float> depthSensor1Scale("ui.depth1 scale",1e-3,8e-4,1e-3);
  //pangolin::Var<float> depthSensor2Scale("ui.depth2 scale",1e-3,8e-4,1e-3);
  //pangolin::Var<float> depthSensor3Scale("ui.depth3 scale",1e-3,8e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> useRgbCamParasForDepth("ui.use rgb cams", true, true);

  //pangolin::Var<float> cam1fu("ui.cam1 fu",rig.cams_[1].params_(0),500,600);
  //pangolin::Var<float> cam1fv("ui.cam1 fv",rig.cams_[1].params_(1),500,600);
  //pangolin::Var<float> cam3fu("ui.cam3 fu",rig.cams_[3].params_(0),500,600);
  //pangolin::Var<float> cam3fv("ui.cam3 fv",rig.cams_[3].params_(1),500,600);
  //pangolin::Var<float> cam5fu("ui.cam5 fu",rig.cams_[5].params_(0),500,600);
  //pangolin::Var<float> cam5fv("ui.cam5 fv",rig.cams_[5].params_(1),500,600);
  //pangolin::Var<float> cam3tx("ui.cam3 tx",rig.T_rcs_[3].translation()(0),0,0.1);
  //pangolin::Var<float> cam3ty("ui.cam3 ty",rig.T_rcs_[3].translation()(1),0,0.1);
  //pangolin::Var<float> cam3tz("ui.cam3 tz",rig.T_rcs_[3].translation()(2),0,0.1);

  pangolin::Var<bool> renderFisheye("ui.fisheye", true, true);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool> fuseTSDF("ui.fuse TSDF",true,true);
  pangolin::Var<float> tsdfMu("ui.mu",0.5,0.,1.);
  pangolin::Var<float> grid0x("ui.grid0 x",-3.0,-2,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-3.0,-2,0);
  pangolin::Var<float> grid0z("ui.grid0 z",0.,0.,1);
  pangolin::Var<float> gridEx("ui.gridE x",3.0,2,0);
  pangolin::Var<float> gridEy("ui.gridE y",3.0,2,0);
  pangolin::Var<float> gridEz("ui.gridE z",3.5,2.,3);

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",7,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",0,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",0,0,10);

  pangolin::Var<bool> dispEst("ui.disp Est", false,true);

  pangolin::RegisterKeyPressCallback('c', [&](){
      for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
      int cId = rgbdStream2cam[sId];
      std::stringstream ss;
      ss << "capture_cam" << cId << ".png";
      try{
      pangolin::SaveImage(
        gui.images[gui.iRGB[sId]], gui.video.Streams()[gui.iRGB[sId]].PixFormat(),
        pangolin::MakeUniqueFilename(ss.str())
        );
      }catch(std::exception e){
      pango_print_error("Unable to save frame: %s\n", e.what());
      }
      }
      });

  // TODO: figure out why removing this will crash my computer...
  tdp::ArucoDetector detector(0.158);

  pangolin::GlRenderBuffer glRenderBuf(w,h);
  pangolin::GlTexture tex(w,h,GL_RGBA8);
  pangolin::GlFramebuffer glFrameBuf(tex, glRenderBuf);
  tdp::ManagedHostImage<tdp::Vector3bda> rgbJoint(w,h);
  memset(rgbJoint.ptr_, 0, rgbJoint.SizeBytes());
  tdp::ManagedHostImage<float> dJoint(w,h);

  pangolin::GlSlProgram colorPc;
  colorPc.AddShaderFromFile(pangolin::GlSlVertexShader,
      "/home/jstraub/workspace/research/tdp/shaders/surround3D.vert");
  colorPc.AddShaderFromFile(pangolin::GlSlFragmentShader,
      "/home/jstraub/workspace/research/tdp/shaders/surround3D.frag");
  colorPc.Link();

  tdp::SE3f T_mr;

  size_t numFused = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    //if (cam1fu.GuiChanged()) rig.cams_[1].params_(0) = cam1fu;
    //if (cam1fv.GuiChanged()) rig.cams_[1].params_(1) = cam1fv;
    //if (cam3fu.GuiChanged()) rig.cams_[3].params_(0) = cam3fu;
    //if (cam3fv.GuiChanged()) rig.cams_[3].params_(1) = cam3fv;
    //if (cam5fu.GuiChanged()) rig.cams_[5].params_(0) = cam5fu;
    //if (cam5fv.GuiChanged()) rig.cams_[5].params_(1) = cam5fv;
    //if (cam3tx.GuiChanged()) rig.T_rcs_[3].matrix()(0,3) = cam3tx;
    //if (cam3ty.GuiChanged()) rig.T_rcs_[3].matrix()(1,3) = cam3ty;
    //if (cam3tz.GuiChanged()) rig.T_rcs_[3].matrix()(2,3) = cam3tz;

    tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
    tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
    tdp::Vector3fda dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    TICK("rgb collection");
    // get rgb image
    for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
      tdp::Image<tdp::Vector3bda> rgbStream;
      if (!gui.ImageRGB(rgbStream, sId)) continue;
      int32_t cId = rgbdStream2cam[sId]; 
      // Get ROI
      tdp::Image<tdp::Vector3bda> rgb_i(wSingle, hSingle,
          rgb.ptr_+cId*rgbStream.Area());
      rgb_i.CopyFrom(rgbStream,cudaMemcpyHostToHost);
    }
    TOCK("rgb collection");
    TICK("depth collection");
    // get depth image
    for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
      tdp::Image<uint16_t> dStream;
      if (!gui.ImageD(dStream, sId)) continue;
      int32_t cId = rgbdStream2cam[sId]; 
      //std::cout << sId << " " << cId << std::endl;
      // Get ROI
      tdp::Image<uint16_t> cuDraw_i(wSingle, hSingle,
          cuDraw.ptr_+cId*dStream.Area());
      cuDraw_i.CopyFrom(dStream,cudaMemcpyHostToDevice);
      // convert depth image from uint16_t to float [m]
      tdp::Image<float> cuD_i(wSingle, hSingle,
          cuD.ptr_+cId*dStream.Area());
      //float depthSensorScale = depthSensor1Scale;
      //if (cId==1) depthSensorScale = depthSensor2Scale;
      //if (cId==2) depthSensorScale = depthSensor3Scale;
      if (rig.depthScales_.size() > cId) {
        float a = rig.scaleVsDepths_[cId](0);
        float b = rig.scaleVsDepths_[cId](1);
        // TODO: dont need to load this every time
        cuScale.CopyFrom(rig.depthScales_[cId],cudaMemcpyHostToDevice);
        tdp::ConvertDepthGpu(cuDraw_i, cuD_i, cuScale, a, b, dMin, dMax);
      //} else {
      //  tdp::ConvertDepthGpu(cuDraw_i, cuD_i, depthSensorScale, dMin, dMax);
      }
    }
    TOCK("depth collection");
    TICK("pc and normals");
    // convert depth image from uint16_t to float [m]
    //tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    // compute point cloud (on CPU)
    for (size_t sId=0; sId < dStream2cam.size(); sId++) {
      int32_t cId;
      if (useRgbCamParasForDepth) {
        cId = rgbStream2cam[sId]; 
      } else {
        cId = dStream2cam[sId]; 
      }
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      tdp::Image<tdp::Vector3fda> cuN_i(wSingle, hSingle,
          cuN.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);
      tdp::Image<tdp::Vector3fda> cuPc_i(wSingle, hSingle,
          cuPc.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);
      tdp::Image<float> cuD_i(wSingle, hSingle,
          cuD.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);

      // compute point cloud from depth in rig coordinate system
      tdp::Depth2PCGpu(cuD_i,cam,T_rc,cuPc_i);
      // compute normals from depth in rig coordinate system
      tdp::Depth2Normals(cuD_i, cam, T_rc.rotation(), cuN_i);
    }
    TOCK("pc and normals");
    TICK("Ray Trace TSDF");
    tdp::Image<tdp::Vector3fda> cuNEst = ns_m.GetImage(0);
    tdp::Image<tdp::Vector3fda> cuPcEst = pcs_m.GetImage(0);
    for (size_t sId=0; sId < dStream2cam.size(); sId++) {
      int32_t cId;
      if (useRgbCamParasForDepth) {
        cId = rgbStream2cam[sId]; 
      } else {
        cId = dStream2cam[sId]; 
      }
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];
      tdp::SE3f T_mo = T_mr*T_rc;

      tdp::Image<tdp::Vector3fda> cuNEst_i(wSingle, hSingle,
          cuNEst.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);
      tdp::Image<tdp::Vector3fda> cuPcEst_i(wSingle, hSingle,
          cuPcEst.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);

      // ray trace the TSDF to get pc and normals in camera cosy
      RayTraceTSDF(cuTSDF, cuPcEst_i, 
          cuNEst_i, T_mo, cam, grid0, dGrid, tsdfMu); 
      // transform pc and normals into rig cosy for ICP
      tdp::TransformPc(T_rc, cuPcEst_i);
      tdp::TransformPc(T_rc.rotation(), cuNEst_i);
    }
    TOCK("Ray Trace TSDF");
    TICK("Setup Pyramids");
    // TODO might want to use the pyramid construction with smoothing
//    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
//        cudaMemcpyDeviceToDevice, 0.03);
    pcs_o.GetImage(0).CopyFrom(cuPc, cudaMemcpyDeviceToDevice);
    tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_o,cudaMemcpyDeviceToDevice);
    tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_m,cudaMemcpyDeviceToDevice);

    ns_o.GetImage(0).CopyFrom(cuN, cudaMemcpyDeviceToDevice);
    tdp::CompleteNormalPyramid<3>(ns_o,cudaMemcpyDeviceToDevice);
    // just complete the surface normals obtained from the TSDF
    tdp::CompleteNormalPyramid<3>(ns_m,cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

    tdp::SE3f dT_mo;
    if (runICP && numFused > 30) {
      if (gui.verbose) std::cout << "icp" << std::endl;
      TICK("ICP");
      std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};

      { // ICP
        size_t lvls = maxIt.size();
        for (int lvl=lvls-1; lvl >= 0; --lvl) {
          float errPrev = 0.f; 
          float error = 0.f; 
          for (size_t it=0; it<maxIt[lvl]; ++it) {
            float count = 0.f; 
            float error = 0.f; 
            Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA;
            Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb;
            ATA.fill(0.);
            ATb.fill(0.);
            tdp::Image<tdp::Vector3fda> pc_ml = pcs_m.GetImage(lvl);
            tdp::Image<tdp::Vector3fda> n_ml = ns_m.GetImage(lvl);
            tdp::Image<tdp::Vector3fda> pc_ol = pcs_o.GetImage(lvl);
            tdp::Image<tdp::Vector3fda> n_ol = ns_o.GetImage(lvl);
            size_t w_l = pc_ml.w_;
            size_t h_l = pc_ml.h_/3;
            for (size_t sId=0; sId < dStream2cam.size(); sId++) {
              int32_t cId;
              if (useRgbCamParasForDepth) {
                cId = rgbStream2cam[sId]; 
              } else {
                cId = dStream2cam[sId]; 
              }
              CameraT cam = rig.cams_[cId];
              tdp::SE3f T_cr = rig.T_rcs_[cId].Inverse();

              // all PC and normals are in rig coordinates
              tdp::Image<tdp::Vector3fda> pc_mli(w_l, h_l,
                  pc_ml.ptr_+rgbdStream2cam[sId]*w_l*h_l);
              tdp::Image<tdp::Vector3fda> pc_oli(w_l, h_l,
                  pc_ol.ptr_+rgbdStream2cam[sId]*w_l*h_l);
              tdp::Image<tdp::Vector3fda> n_mli(w_l, h_l,
                  n_ml.ptr_+rgbdStream2cam[sId]*w_l*h_l);
              tdp::Image<tdp::Vector3fda> n_oli(w_l, h_l,
                  n_ol.ptr_+rgbdStream2cam[sId]*w_l*h_l);

              Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA_i;
              Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb_i;
              float error_i = 0;
              float count_i = 0;
              // Compute ATA and ATb from A x = b
              ICPStep(pc_mli, n_mli, pc_oli, n_oli,
                  dT_mo, T_cr, tdp::ScaleCamera<float>(cam,pow(0.5,lvl)),
                  cos(icpAngleThr_deg*M_PI/180.),
                  icpDistThr,ATA_i,ATb_i,error_i,count_i);
              ATA += ATA_i;
              ATb += ATb_i;
              error += error_i;
              count += count_i;
            }
            if (count < 100) {
              std::cout << "# inliers " << count << " to small " << std::endl;
              break;
            }
            // solve for x using ldlt
            Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
              (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 
            // apply x to the transformation
            dT_mo = tdp::SE3f(tdp::SE3f::Exp_(x))*dT_mo;
            std::cout << "lvl " << lvl << " it " << it 
              << ": err=" << error << "\tdErr/err=" << fabs(error-errPrev)/error
              << " # inliers: " << count 
              //<< " |ATA|=" << ATA.determinant()
              //<< " x=" << x.transpose()
              << std::endl;
            //std::cout << dT.matrix() << std::endl;
            //std::cout << T_mo.matrix() << std::endl;
            if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
            errPrev = error;
          }
        }
      }

//      tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_o, ns_o, dT,
//          camD, maxIt, icpAngleThr_deg, icpDistThr); 
//      std::cout << dT.matrix3x4() << std::endl;
      T_mr = dT_mo*T_mr;
//      //std::cout << "T_mo" << std::endl << T_mo.matrix3x4() << std::endl;
      TOCK("ICP");
    }

    if (pangolin::Pushed(resetTSDF)) {
      T_mr.matrix() = Eigen::Matrix4f::Identity();
      W.Fill(0.);
      TSDF.Fill(-1.01);
      tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
      tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);
      numFused = 0;
    }
    if (pangolin::Pushed(resetICP)) {
      T_mr.matrix() = Eigen::Matrix4f::Identity();
    }

    if (fuseTSDF || numFused <= 30) {
      if (gui.verbose) std::cout << "add to tsdf" << std::endl;
      TICK("Add To TSDF");
      for (size_t sId=0; sId < dStream2cam.size(); sId++) {
        int32_t cId;
        if (useRgbCamParasForDepth) {
          cId = rgbStream2cam[sId]; 
        } else {
          cId = dStream2cam[sId]; 
        }
        CameraT cam = rig.cams_[cId];
        tdp::SE3f T_rc = rig.T_rcs_[cId];
        tdp::SE3f T_mo = T_mr+T_rc;
        tdp::Image<float> cuD_i(wSingle, hSingle,
            cuD.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);
        AddToTSDF(cuTSDF, cuW, cuD_i, T_mo, cam, grid0, dGrid, tsdfMu); 
      }
      numFused ++;
      TOCK("Add To TSDF");
    }

    // Draw 3D stuff
    if (d_cam.IsShown()) {
      if (dispEst) {
        pc.CopyFrom(pcs_m.GetImage(0),cudaMemcpyDeviceToHost);
      } else {
        pc.CopyFrom(pcs_o.GetImage(0),cudaMemcpyDeviceToHost);
      }
      glEnable(GL_DEPTH_TEST);
      d_cam.Activate(s_cam);
      // draw the axis
      pangolin::glDrawAxis(0.1);

      vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
      // render point cloud
      pangolin::RenderVboCbo(vbo,cbo,true);
      glDisable(GL_DEPTH_TEST);
    }

    if (renderFisheye) {
      glFrameBuf.Bind();

      colorPc.Bind();

      glViewport(0, 0, w, h);
      glClearColor(0,0,0,1);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
      glColor3f(1.0f, 1.0f, 1.0f);

      Eigen::Matrix4f MV = s_cam.GetModelViewMatrix();
      Eigen::Matrix4f P = pangolin::ProjectionMatrix(640,3*480,420,420,320,3*240,0.1,1000);
      colorPc.SetUniform("P", P);
      colorPc.SetUniform("MV", MV);

      size_t stride = sizeof(float)*3+sizeof(uint8_t)*3;
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      vbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
      cbo.Bind();
      glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, NULL);
      pangolin::RenderVbo(vbo);
      glDisableVertexAttribArray(0);
      glDisableVertexAttribArray(1);

      cbo.Unbind();
      vbo.Unbind();

      colorPc.Unbind();

      glFrameBuf.Unbind();

      if (viewRgbJoint.IsShown()) {
        tex.Download(rgbJoint.ptr_, GL_RGB, GL_UNSIGNED_BYTE);
        viewRgbJoint.SetImage(rgbJoint);
      }
    }

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
}


int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";

  if( argc > 1 ) {
    const std::string input_uri = std::string(argv[1]);
    const std::string configPath = (argc > 2) ? std::string(argv[2]) : "../config/surround3D_2016_09_11.json";
    try{
      VideoViewer(input_uri, configPath, dflt_output_uri);
    } catch (pangolin::VideoException e) {
      std::cout << e.what() << std::endl;
    }
  }else{
    const std::string input_uris[] = {
      "dc1394:[fps=30,dma=10,size=640x480,iso=400]//0",
      "convert:[fmt=RGB24]//v4l:///dev/video0",
      "convert:[fmt=RGB24]//v4l:///dev/video1",
      "openni:[img1=rgb]//",
      "test:[size=160x120,n=1,fmt=RGB24]//"
        ""
    };

    std::cout << "Usage  : VideoViewer [video-uri]" << std::endl << std::endl;
    std::cout << "Where video-uri describes a stream or file resource, e.g." << std::endl;
    std::cout << "\tfile:[realtime=1]///home/user/video/movie.pvn" << std::endl;
    std::cout << "\tfile:///home/user/video/movie.avi" << std::endl;
    std::cout << "\tfiles:///home/user/seqiemce/foo%03d.jpeg" << std::endl;
    std::cout << "\tdc1394:[fmt=RGB24,size=640x480,fps=30,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tdc1394:[fmt=FORMAT7_1,size=640x480,pos=2+2,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tv4l:///dev/video0" << std::endl;
    std::cout << "\tconvert:[fmt=RGB24]//v4l:///dev/video0" << std::endl;
    std::cout << "\tmjpeg://http://127.0.0.1/?action=stream" << std::endl;
    std::cout << "\topenni:[img1=rgb]//" << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
