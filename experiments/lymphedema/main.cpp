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
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image_io.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/gui/gui_base.hpp>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SE3.h>
#include <tdp/gui/gui.hpp>
#include <tdp/camera/camera_poly.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/icp/icp.h>
#include <tdp/gl/render.h>

#include <tdp/utils/threadedValue.hpp>
#include <tdp/gl/shaders.h>
#include <tdp/marching_cubes/marching_cubes.h>

#include <pangolin/video/drivers/realsense.h>

typedef tdp::CameraPoly3<float> CameraT;
//typedef tdp::Camera<float> CameraT;

int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  std::string input_uri = std::string(argv[1]);
  std::string configPath = std::string(argv[2]);
  std::string output_uri = (argc > 3) ? std::string(argv[3]) : dflt_output_uri;
  std::string tsdfOutputPath = "tsdf.raw";

  std::cout << input_uri << std::endl;
  std::cout << configPath << std::endl;

  // Read rig file
  tdp::Rig<CameraT> rig;
  if (!rig.FromFile(configPath, true)) {
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

  tdp::GuiBase gui(1200,800,video);
  Stopwatch::getInstance().setCustomSignature(1237249817410);

  size_t wSingle = video.Streams()[0].Width();
  size_t hSingle = video.Streams()[0].Height();
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  wSingle += wSingle%64;
  hSingle += hSingle%64;
  size_t w = wSingle;
  size_t h = rig.NumCams()*hSingle;

//  size_t dTSDF = 128;
//  size_t wTSDF = 128;
//  size_t hTSDF = 128;

//  size_t dTSDF = 512;
//  size_t wTSDF = 512;
//  size_t hTSDF = 512;
  size_t dTSDF = 256;
  size_t wTSDF = 256;
  size_t hTSDF = 256;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);
  pangolin::View& viewMeshOut = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewMeshOut);
  // add a simple image viewer
  tdp::QuickView viewRgb(w,h);
  gui.container().AddDisplay(viewRgb);
  tdp::QuickView viewD(w,h);
  gui.container().AddDisplay(viewD);
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);

  tdp::QuickView viewDebug1(640,480);
  gui.container().AddDisplay(viewDebug1);
  tdp::QuickView viewDebug2(640,480);
  gui.container().AddDisplay(viewDebug2);

  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> n(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPc(w, h);

  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_o(w,h);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_o(w,h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  tdp::ManagedHostVolume<tdp::TSDFval> TSDF(wTSDF, hTSDF, dTSDF);
  TSDF.Fill(tdp::TSDFval(-1.01,0.));
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTSDF(wTSDF, hTSDF, dTSDF);
  cuTSDF.CopyFrom(TSDF);
  pangolin::GlBuffer meshVbo;
  pangolin::GlBuffer meshCbo;
  pangolin::GlBuffer meshIbo;

  // Add some variables to GUI
  pangolin::Var<float> dMin("ui.d min",0.01,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",0.5,0.1,4.);

  pangolin::Var<int> ir("ui.IR", 16,0,16);
//  pangolin::Var<bool> grabOneFrame("ui.grabOneFrame", true, false);
  pangolin::Var<bool> rotatingDepthScan("ui.rotating scan", false, true);
  pangolin::Var<int> rotatingDepthScanIrPower("ui.IR power", 16,0,16);
  pangolin::Var<int> stabilizationTime("ui.stabil. dt ms", 30, 1, 100);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool>  saveTSDF("ui.save TSDF", false, false);
  pangolin::Var<bool> fuseTSDF("ui.fuse TSDF",true,true);
  pangolin::Var<float> tsdfMu("ui.mu",0.002,0.0001,0.01);
  pangolin::Var<float> tsdfWThr("ui.w thr",3,1.,20.);
  pangolin::Var<float> tsdfWMax("ui.w max",100.,1.,300.);
  pangolin::Var<float> grid0x("ui.grid0 x",-0.175,-.5,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-0.116,-.5,0);
  pangolin::Var<float> grid0z("ui.grid0 z",0.320,0.,0.3);
  pangolin::Var<float> gridEx("ui.gridE x",0.18,0.5,0.);
  pangolin::Var<float> gridEy("ui.gridE y",0.074,0.5,0.);
  pangolin::Var<float> gridEz("ui.gridE z",0.500,0.9,0.);
  pangolin::Var<int>   pyrLvl("ui.pyr lvl disp",1,0,2);

  pangolin::Var<bool>   useRgbCamParasForDepth("ui.use rgb cams", true, true);
  pangolin::Var<bool>  runICP("ui.run ICP", false, true);
  pangolin::Var<bool>  alignIndividual("ui.individual ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.03,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,20);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,20);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,20);

  pangolin::Var<bool>  runMarchingCubes("ui.run Marching Cubes", false, false);
  pangolin::Var<float> marchCubesfThr("ui.f Thr", 1.0,0.,1.);
  pangolin::Var<float> marchCubeswThr("ui.weight Thr", 0,0,10);

  pangolin::Var<bool>  showPc("ui.showPc", true, true);
  pangolin::Var<bool>  showBox("ui.showBox", true, true);
  pangolin::Var<bool>  showAxes("ui.showAxes", true, true);
  pangolin::Var<bool>  showMesh("ui.showMesh", true, true);
  pangolin::Var<bool> dispEst("ui.disp Est", true,true);

  tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
  tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
  tdp::Vector3fda dGrid = gridE - grid0;

  tdp::SE3f T_mr;
  tdp::SE3f T_wG;

  tdp::ThreadedValue<bool> runSave(true);
  std::thread workThread([&]() {
        while(runSave.Get()) {
          if (pangolin::Pushed(saveTSDF)) {
            TSDF.CopyFrom(cuTSDF);
            std::cout << "start writing TSDF to " << tsdfOutputPath << std::endl;
            tdp::TSDF::SaveTSDF(TSDF, grid0, dGrid, T_wG, tsdfOutputPath);
            std::cout << "done writing TSDF to " << tsdfOutputPath << std::endl;
          }
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      });

  pangolin::RealSenseVideo* rs = video.Cast<pangolin::RealSenseVideo>();
  uint8_t buffer[640*480*(2+rig.NumCams())];

  tdp::Image<uint16_t> _d(640,480,(uint16_t*)buffer,tdp::Storage::Cpu);
  tdp::Image<tdp::Vector3bda> _rgb(640,480,(tdp::Vector3bda*)&buffer[640*480*2],tdp::Storage::Cpu);

  tdp::ThreadedValue<bool> received(true);
  std::thread* threadCollect = nullptr;

  std::vector<tdp::SE3f> T_rcs0 = rig.T_rcs_;

  gui.verbose = true;
  size_t numFrames = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    grid0 << grid0x,grid0y,grid0z;
    gridE << gridEx,gridEy,gridEz;
    dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    if (ir.GuiChanged()) {
      rs->SetPowers(ir);
    }

    if (rotatingDepthScan.GuiChanged() && rotatingDepthScan) {
      rs->SetPowers(0);
      numFrames = 0;
      received.Set(true);
      resetTSDF = true;
      runICP = true;
    }
    if (rotatingDepthScan.GuiChanged() && !rotatingDepthScan) {
      rs->SetPowers(ir);
      numFrames = 0;
      received.Set(true);
      resetTSDF = true;
    }

    if (pangolin::Pushed(resetTSDF)) {
      T_mr = tdp::SE3f(); 
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      cuTSDF.CopyFrom(TSDF);
      numFrames = 0;
    }

    if (pangolin::Pushed(runMarchingCubes)) {
      TSDF.CopyFrom(cuTSDF);
      tdp::ComputeMesh(TSDF, grid0, dGrid,
          T_wG, meshVbo, meshCbo, meshIbo, marchCubeswThr, marchCubesfThr);      
    }

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (rotatingDepthScan) {
      // start a collection thread to do the work so the rendering is
      // smooth
      if (received.Get()) {
        if (threadCollect) {
          threadCollect->join();
          delete threadCollect;
          threadCollect = nullptr;
        }
        threadCollect = new std::thread([&](){
//          TICK("rgbd collection");
          cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
          for (size_t sId=0; sId < rig.rgbdStream2cam_.size(); sId++) {
            // grab one frame 
            rs->SetPower(sId, rotatingDepthScanIrPower);
            std::this_thread::sleep_for (std::chrono::milliseconds(stabilizationTime));
            rs->GrabOne(sId, buffer);
            rs->SetPower(sId, 0);
            int32_t cId = rig.rgbdStream2cam_[sId]; 

            tdp::Image<tdp::Vector3bda> rgb_i = rgb.GetRoi(0,cId*hSingle, wSingle, hSingle);
            tdp::Image<uint16_t> cuDraw_i = cuDraw.GetRoi(0,cId*hSingle, wSingle, hSingle);
            rgb_i.CopyFrom(_rgb);
            cuDraw_i.CopyFrom(_d);
            // convert depth image from uint16_t to float [m]
            tdp::Image<float> cuD_i = cuD.GetRoi(0, cId*hSingle, wSingle, hSingle);
            if (rig.cuDepthScales_.size() > cId) {
              tdp::ConvertDepthGpu(cuDraw_i, cuD_i, rig.cuDepthScales_[cId], 
                  rig.scaleVsDepths_[cId](0), rig.scaleVsDepths_[cId](1), dMin, dMax);
            } else if (rig.depthSensorUniformScale_.size() > cId) {
              tdp::ConvertDepthGpu(cuDraw_i, cuD_i, rig.depthSensorUniformScale_[cId], dMin, dMax);
            } else {
              std::cout << "Warning no scale information found" << std::endl;
            }
          }
//          TOCK("rgbd collection");
          received.Set(true);
        });
        std::cout << "received" << std::endl;
        ++numFrames;
      }
    } else {
      // get next frames from the video source
      gui.NextFrames();
      TICK("rgb collection");
      rig.CollectRGB(gui, rgb);
      TOCK("rgb collection");
      TICK("depth collection");
      int64_t t_host_us_d = 0;
      cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
      rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
      TOCK("depth collection");
      received.Set(true);
      ++numFrames;
    }
    
    if (received.Get()) {
      cuRgb.CopyFrom(rgb);
      TICK("pc and normals");
      rig.ComputePc(cuD, useRgbCamParasForDepth, cuPc);
      rig.ComputeNormals(cuD, useRgbCamParasForDepth, cuN);
      TOCK("pc and normals");
      TICK("Setup Pyramids");
      pcs_o.GetImage(0).CopyFrom(cuPc);
      ns_o.GetImage(0).CopyFrom(cuN);
      tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_o);
      tdp::CompleteNormalPyramid<3>(ns_o);
      TOCK("Setup Pyramids");

      if (!gui.paused() && fuseTSDF ) {
        if (runICP && numFrames > 1) {
          std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
          std::vector<float> errPerLvl;
          std::vector<float> countPerLvl;
          Eigen::Matrix<float,6,6> Sigma_mr; 

          if (alignIndividual) {
            tdp::ICP::ComputeProjectiveUpdateIndividual<CameraT>(
                pcs_m, ns_m, pcs_o, ns_o,
                rig, rig.rgbStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
                gui.verbose, T_mr, errPerLvl, countPerLvl);
          } else {
            // reset to previous value - maybe not wanted/needed?
            rig.T_rcs_ = T_rcs0; 
            if (useRgbCamParasForDepth) {
              tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_o, ns_o,
                  rig, rig.rgbStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
                  gui.verbose, T_mr, Sigma_mr, errPerLvl, countPerLvl);
            } else {
              tdp::ICP::ComputeProjective<CameraT>(pcs_m, ns_m, pcs_o, ns_o,
                  rig, rig.dStream2cam_, maxIt, icpAngleThr_deg, icpDistThr,
                  gui.verbose, T_mr, Sigma_mr, errPerLvl, countPerLvl);
            }
          }
        }
        //    	std::cout << "fusing a frame" << std::endl;
        TICK("Add To TSDF");
        rig.AddToTSDF(cuD, cuRgb, T_mr, useRgbCamParasForDepth, 
            grid0, dGrid, tsdfMu, tsdfWMax, cuTSDF);
        TOCK("Add To TSDF");
        TICK("Ray Trace TSDF");
        rig.RayTraceTSDF(cuTSDF, T_mr, useRgbCamParasForDepth, grid0,
            dGrid, tsdfMu, tsdfWThr, pcs_m, ns_m);
        TOCK("Ray Trace TSDF");
      }
			received.Set(false);
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);
      // draw the axis
      if (showAxes) {
        for (size_t i=0; i<rig.cams_.size(); ++i) {
          auto& T = rig.T_rcs_[i];
          auto& cam = rig.cams_[i];
          pangolin::glDrawAxis(T.matrix(), 0.1f);
          pangolin::glDrawFrustrum(cam.GetKinv(), wSingle, hSingle, T.matrix(), 0.1f);
        }
      }
      if (showBox) {
        Eigen::AlignedBox3f box(grid0,gridE);
        glColor4f(1,0,0,0.5f);
        pangolin::glDrawAlignedBox(box);
      }
      if (showPc) {
        pc.CopyFrom(cuPc, cudaMemcpyDeviceToHost);
        vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
        cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
        // render point cloud
        pangolin::RenderVboCbo(vbo,cbo,true);
        pc.CopyFrom(pcs_m.GetImage(pyrLvl), cudaMemcpyDeviceToHost);
        vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
        glColor3f(0,1,0);
        pangolin::RenderVbo(vbo);
      }
      if (showMesh && meshVbo.num_elements > 0
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
    if (viewMeshOut.IsShown()) {
      viewMeshOut.Activate(s_cam);
      tdp::RenderVboIboCbo(meshVbo, meshIbo, meshCbo);
    }
    glDisable(GL_DEPTH_TEST);

    // Draw 2D stuff
    //gui.ShowFrames();
    if (viewRgb.IsShown()) {
      viewRgb.SetImage(rgb);
    }
    if (viewD.IsShown()) {
      d.CopyFrom(cuD);
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
    pangolin::FinishFrame();
  }

  runSave.Set(false);
  workThread.join();
}
