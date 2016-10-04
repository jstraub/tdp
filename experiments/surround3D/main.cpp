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
  std::vector<int32_t> rgbStream2cam;
  std::vector<int32_t> dStream2cam;
  std::vector<int32_t> rgbdStream2cam;
  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  tdp::CorrespondOpenniStreams2Cams(streams,rig,rgbStream2cam,
      dStream2cam, rgbdStream2cam);

  // optionally connect to IMU if it is found.
  tdp::ImuInterface* imu = nullptr; 
  if (imu_input_uri.size() > 0) 
    imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();
  tdp::ImuInterpolator imuInterp(imu,nullptr);
  imuInterp.Start();

  tdp::GuiBase gui(1200,800,video);

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

  tdp::Camera<float> camView(Eigen::Vector4f(220,220,319.5,239.5)); 
  tdp::ManagedDeviceImage<float> cuDView(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcView(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> nEstdummy(w,h);
  
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

  tdp::ManagedHostVolume<tdp::TSDFval> TSDF(wTSDF, hTSDF, dTSDF);
  TSDF.Fill(tdp::TSDFval(-1.01,0.));
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTSDF(wTSDF, hTSDF, dTSDF);
  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> useRgbCamParasForDepth("ui.use rgb cams", true, true);

  pangolin::Var<bool> renderFisheye("ui.fisheye", false, true);

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

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

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

  tdp::SE3f T_mr;
  size_t numFused = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {

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
      tdp::Image<tdp::Vector3bda> rgb_i = rgb.GetRoi(0,cId*hSingle,
          wSingle, hSingle);
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
      tdp::Image<uint16_t> cuDraw_i = cuDraw.GetRoi(0,cId*hSingle,
          wSingle, hSingle);
      cuDraw_i.CopyFrom(dStream,cudaMemcpyHostToDevice);
      // convert depth image from uint16_t to float [m]
      tdp::Image<float> cuD_i = cuD.GetRoi(0, cId*hSingle, 
          wSingle, hSingle);
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
    for (size_t sId=0; sId < dStream2cam.size(); sId++) {
      int32_t cId;
      if (useRgbCamParasForDepth) {
        cId = rgbStream2cam[sId]; 
      } else {
        cId = dStream2cam[sId]; 
      }
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      tdp::Image<tdp::Vector3fda> cuN_i = cuN.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);
      tdp::Image<tdp::Vector3fda> cuPc_i = cuPc.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);
      tdp::Image<float> cuD_i = cuD.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);

      // compute point cloud from depth in rig coordinate system
      tdp::Depth2PCGpu(cuD_i, cam, T_rc, cuPc_i);
      // compute normals from depth in rig coordinate system
      tdp::Depth2Normals(cuD_i, cam, T_rc.rotation(), cuN_i);
    }
    TOCK("pc and normals");
    if (gui.verbose) std::cout << "ray trace tsdf" << std::endl;
    TICK("Setup Pyramids");
    // TODO might want to use the pyramid construction with smoothing
//    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
//        cudaMemcpyDeviceToDevice, 0.03);
    pcs_o.GetImage(0).CopyFrom(cuPc, cudaMemcpyDeviceToDevice);
    tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_o,cudaMemcpyDeviceToDevice);

    ns_o.GetImage(0).CopyFrom(cuN, cudaMemcpyDeviceToDevice);
    tdp::CompleteNormalPyramid<3>(ns_o,cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

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
              tdp::Image<tdp::Vector3fda> pc_mli = pc_ml.GetRoi(0,
                  rgbdStream2cam[sId]*h_l, w_l, h_l);
              tdp::Image<tdp::Vector3fda> pc_oli = pc_ol.GetRoi(0,
                  rgbdStream2cam[sId]*h_l, w_l, h_l);
              tdp::Image<tdp::Vector3fda> n_mli = n_ml.GetRoi(0,
                  rgbdStream2cam[sId]*h_l, w_l, h_l);
              tdp::Image<tdp::Vector3fda> n_oli = n_ol.GetRoi(0,
                  rgbdStream2cam[sId]*h_l, w_l, h_l);

              Eigen::Matrix<float,6,6,Eigen::DontAlign> ATA_i;
              Eigen::Matrix<float,6,1,Eigen::DontAlign> ATb_i;
              float error_i = 0;
              float count_i = 0;
              // Compute ATA and ATb from A x = b
              ICPStep(pc_mli, n_mli, pc_oli, n_oli,
                  T_mr, T_cr, tdp::ScaleCamera<float>(cam,pow(0.5,lvl)),
                  cos(icpAngleThr_deg*M_PI/180.),
                  icpDistThr,ATA_i,ATb_i,error_i,count_i);
              ATA += ATA_i;
              ATb += ATb_i;
              error += error_i;
              count += count_i;
            }
            if (count < 1000) {
              std::cout << "# inliers " << count << " to small " << std::endl;
              break;
            }
            // solve for x using ldlt
            Eigen::Matrix<float,6,1,Eigen::DontAlign> x =
              (ATA.cast<double>().ldlt().solve(ATb.cast<double>())).cast<float>(); 
            // apply x to the transformation
            T_mr = tdp::SE3f(tdp::SE3f::Exp_(x))*T_mr;
            std::cout << "lvl " << lvl << " it " << it 
              << ": err=" << error 
              << "\tdErr/err=" << fabs(error-errPrev)/error
              << " # inliers: " << count 
              //<< " |ATA|=" << ATA.determinant()
              //<< " x=" << x.transpose()
              << std::endl;
            if (it>0 && fabs(error-errPrev)/error < 1e-7) break;
            errPrev = error;
          }
        }
      }
      TOCK("ICP");
    }

    if (pangolin::Pushed(resetTSDF)) {
      T_mr.matrix() = Eigen::Matrix4f::Identity();
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
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
        AddToTSDF(cuTSDF, cuD_i, T_mo, cam, grid0, dGrid, tsdfMu); 
      }
      numFused ++;
      TOCK("Add To TSDF");
    }

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

      tdp::Image<tdp::Vector3fda> cuNEst_i = cuNEst.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);
      tdp::Image<tdp::Vector3fda> cuPcEst_i = cuPcEst.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);

      // ray trace the TSDF to get pc and normals in model cosy
      RayTraceTSDF(cuTSDF, cuPcEst_i, 
          cuNEst_i, T_mo, cam, grid0, dGrid, tsdfMu); 
    }
    // just complete the surface normals obtained from the TSDF
    tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_m,cudaMemcpyDeviceToDevice);
    tdp::CompleteNormalPyramid<3>(ns_m,cudaMemcpyDeviceToDevice);
    TOCK("Ray Trace TSDF");

    // Render point cloud from viewpoint of origin
    tdp::SE3f T_mv;
    T_mv.matrix()(2,3) = -3.;
    RayTraceTSDF(cuTSDF, cuDView, nEstdummy, T_mv, camView, grid0,
        dGrid, tsdfMu); 
    tdp::Depth2PCGpu(cuDView,camView,cuPcView);

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
  saveTSDF = true;

  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;

  std::this_thread::sleep_for(std::chrono::microseconds(500));
  runWorker.Set(false);
  workThread.join();
  return 0;
}

