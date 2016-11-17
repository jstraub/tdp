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
#include <tdp/preproc/normals.h>
#include <tdp/preproc/lab.h>

#include <tdp/gui/gui.hpp>

int main( int argc, char* argv[] )
{
  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

  bool runOnce = false;
  if (!option.compare("-1")) {
    runOnce = true; 
  }

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, "./video.pango");
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  tdp::GUI gui(1200,800,video);

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
  tdp::QuickView viewLab(w,h);
  gui.container().AddDisplay(viewLab);
  tdp::QuickView viewZ(w,h);
  gui.container().AddDisplay(viewZ);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> lab(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> lab8(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> abd(w, h);
  tdp::ManagedHostImage<uint16_t> z(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuAbd(w, h);
  tdp::ManagedDeviceImage<uint16_t> cuZ(w, h);
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);
  pangolin::Var<float> lambda("ui.lambda",0.3,0.01,1.);
  pangolin::Var<int> maxIt("ui.max it",10,1,100);
  pangolin::Var<int> minNchangePerc("ui.min N change perc",0.05, 0.01, 0.1);

  tdp::DPmeans dpMeans(lambda);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    // get rgb image
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    tdp::Rgb2Lab(rgb, lab);
    // get depth image
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    // copy raw image to gpu
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    // convet depth image from uint16_t to float [m]
    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);

    for (size_t i=0; i<abd.Area(); ++i) {
      abd[i](0) = lab[i](1);
      abd[i](1) = lab[i](2);
      abd[i](2) = (d[i]-dMin)/(dMax-dMin);
    }
    cuAbd.CopyFrom(abd, cudaMemcpyHostToDevice);
    dpMeans.lambda_ = lambda;
    dpMeans.Compute(abd, cuAbd, cuZ, maxIt, minNchangePerc);

    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();

    if (viewLab.IsShown()) {
      tdp::Rgb2Lab(rgb, lab8);
      viewLab.SetImage(lab8);
    }
    if (viewZ.IsShown()) {
      z.CopyFrom(cuZ, cudaMemcpyDeviceToHost);
      viewZ.SetImage(z);
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


