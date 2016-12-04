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
#include <pangolin/video/drivers/realsense.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/preproc/lab.h>
#include <tdp/camera/camera.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/normals.h>
#include <tdp/utils/timer.hpp>

#include <tdp/gui/gui.hpp>
#include <tdp/io/tinyply.h>

int main( int argc, char* argv[] )
{
  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, "./video.pango");
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  tdp::GuiBase gui(1200,800,video);

  size_t w = video.Streams()[gui.iD[0]].Width();
  size_t h = video.Streams()[gui.iD[0]].Height();
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  w += w%64;
  h += h%64;

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

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> n(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> lab(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-4,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);
  pangolin::Var<float> colorDiff("ui.color diff",10.,1.0,40.);
  pangolin::Var<bool> binaryPly("ui.binary ply file",false,true);
  pangolin::Var<bool> savePC("ui.save current PC",false,false);

  pangolin::Var<bool> rotatingScan("ui.rotating Scan",true,false);

  pangolin::Var<int> stabilizationTime("ui.stabil. dt ms", 30, 1, 30);

  pangolin::RealSenseVideo* rs = video.Cast<pangolin::RealSenseVideo>();
  uint8_t buffer[640*480*(2+10)];

  tdp::Image<uint16_t> _d(640,480,(uint16_t*)buffer,tdp::Storage::Cpu);
  tdp::Image<tdp::Vector3bda> _rgb(640,480,
      (tdp::Vector3bda*)&buffer[640*480*2],tdp::Storage::Cpu);

  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<uint16_t> dRaw(w, h);

  rs->SetPowers(0);
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    if (rotatingScan.GuiChanged() && !rotatingScan) {
      rs->SetPower(0, 16);
    }
    
    if (!rotatingScan) {
      gui.NextFrames();
      // get rgb image
      tdp::Image<tdp::Vector3bda> _rgb;
      if (!gui.ImageRGB(rgb)) continue;
      // get depth image
      tdp::Image<uint16_t> _dRaw;
      if (!gui.ImageD(dRaw)) continue;
      rgb.CopyFrom(_rgb);
      dRaw.CopyFrom(_dRaw);
    } else {
      gui.TicToc();
      // grab one frame 
      tdp::Timer t0;
      rs->SetPower(0, 16);
      std::this_thread::sleep_for (std::chrono::milliseconds(stabilizationTime));
      rs->GrabOne(0, buffer);
      rs->SetPower(0, 0);
      t0.toctic("frame capture");

      rgb.CopyFrom(_rgb);
      dRaw.CopyFrom(_d);
    }

    tdp::Rgb2LabCpu(rgb, lab);
    for (size_t u=1; u<pc.w_-1; ++u) {
      for (size_t v=1; v<pc.h_-1; ++v) {
        if (dRaw(u,v) == 0) 
          continue;
        tdp::Vector3fda color = lab(u,v);
        if (dRaw(u-1,v) == 0 || dRaw(u+1,v) == 0) {
          size_t uN = u;
          while(uN < pc.w_ 
              && dRaw(uN,v) != 0 
              && (color-lab(uN++,v)).norm() < colorDiff) {
            rgb(uN-1,v) << 255, 0, 0;
          }
        }
      }
    }

    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    // copy raw image to gpu
    cuDraw.CopyFrom(dRaw);
    // convet depth image from uint16_t to float [m]
    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    d.CopyFrom(cuD);
    // compute point cloud (on CPU)
    tdp::Depth2PC(d,cam,pc);
    // compute normals
    tdp::Depth2Normals(cuD, cam, cuN);
    // convert normals to RGB image
    tdp::Normals2Image(cuN, cuN2D);
    // copy normals image to CPU memory
    n2D.CopyFrom(cuN2D);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
//    gui.ShowFrames();
    // render normals image
    viewRgb.SetImage(rgb);
    viewD.SetImage(dRaw);
    viewN2D.SetImage(n2D);

    // if pressed savePC button save the point cloud to a file.
    if (pangolin::Pushed(savePC)) {
      n.CopyFrom(cuN);
      std::vector<std::string> comments;
      comments.push_back("generated from simpleGui");
      tdp::SavePointCloud(pangolin::MakeUniqueFilename("mesh.ply"),
          pc, n, rgb, binaryPly, comments);
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
  return 0;
}


