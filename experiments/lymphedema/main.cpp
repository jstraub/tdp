/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
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

  tdp::GuiBase gui(1200,800,video);
  Stopwatch::getInstance().setCustomSignature(1237249810);

  size_t wSingle = video.Streams()[0].Width();
  size_t hSingle = video.Streams()[0].Height();
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  wSingle += wSingle%64;
  hSingle += hSingle%64;
  size_t w = wSingle;
  size_t h = 3*hSingle;

  size_t dTSDF = 512;
  size_t wTSDF = 512;
  size_t hTSDF = 512;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);
  // add a simple image viewer
  tdp::QuickView viewRgb(w,h);
  gui.container().AddDisplay(viewRgb);
  tdp::QuickView viewD(w,h);
  gui.container().AddDisplay(viewD);
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);

  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> n(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

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

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-4,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

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

  pangolin::Var<bool> useRgbCamParasForDepth("ui.use rgb cams", true, true);

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

    TICK("Setup Pyramids");
    // TODO might want to use the pyramid construction with smoothing
    pcs_o.GetImage(0).CopyFrom(cuPc, cudaMemcpyDeviceToDevice);
    tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_o,cudaMemcpyDeviceToDevice);
    ns_o.GetImage(0).CopyFrom(cuN, cudaMemcpyDeviceToDevice);
    tdp::CompleteNormalPyramid<3>(ns_o,cudaMemcpyDeviceToDevice);
    TOCK("Setup Pyramids");

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    if (d_cam.IsShown()) {
      d_cam.Activate(s_cam);
      // draw the axis
      for (auto& T : rig.T_rcs_) {
        pangolin::glDrawAxis(T.matrix(), 0.1f);
      }
      Eigen::AlignedBox3f box(grid0,gridE);
      glColor4f(1,0,0,0.5f);
      pangolin::glDrawAlignedBox(box);

      vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
      // render point cloud
      pangolin::RenderVboCbo(vbo,cbo,true);
    }

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    if (viewRgb.IsShown()) {
      viewRgb.SetImage(rgb);
    }
    if (viewD.IsShown()) {
      d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
      viewD.SetImage(d);
    }
    if (viewN2D.IsShown()) {
      // convert normals to RGB image
      tdp::Normals2Image(cuN, cuN2D);
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
}
