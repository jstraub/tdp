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

#include <tdp/gui/gui.hpp>

#include <tdp/data/managed_volume.h>
#include "CIsoSurface.h"
#include <iostream>

int main( int argc, char* argv[] )
{
  std::cout << "Start" << std::endl;
  const std::string dflt_output_uri = "pango://video.pango";
  const std::string input_uri = std::string(argv[1]);
  const std::string output_uri = (argc > 2) ? std::string(argv[2]) : dflt_output_uri;

  std::cout << "begin videorecord" << std::endl;
  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  std::cout << "end videorecord init" << std::endl;
  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  // LOAD TSDF
  // tdp::ManagedHostVolume<TSDFval> tsdf;
  // if (!tdp::LoadVolume(tsdf, "desk0_tsdf.raw")) {
  //  pango_print_error("Unable to load volume");
  //  return 1;
  // }

  float test[27] = {
        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,
        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,
        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f};
  std::cout << "Built test array" << std::endl;
  // procesing of TSDF
  CIsoSurface<float> surface;
  surface.GenerateSurface(test, 0.0f, 4, 4, 4, .02, .02, .02);
  std::cout << "Generated Surface" << std::endl;
  if (!surface.IsSurfaceValid()) {
    pango_print_error("Unable to generate surface");
    return 1;
  }

  float* vertexStore = new float[surface.numVertices()];
  unsigned int* indexStore = new unsigned int[surface.numTriangles() * 3];
  surface.getVertices(vertexStore);
  surface.getIndices(indexStore);

  std::cout << "Number of Vertices: " << surface.numVertices() << std::endl;
  for (size_t i = 0; i < surface.numVertices(); i++) {
    std::cout << vertexStore[i] << " ";
  }

  std::cout << "\nNumber of Triangles: " << surface.numTriangles() << std::endl;
  for (size_t i = 0; i < surface.numTriangles(); i++) {
    std::cout << indexStore[3*i] << " ";
    std::cout << indexStore[3*i + 1] << " ";
    std::cout << indexStore[3*i + 2] << std::endl;
  }

  // have mesh

  tdp::GUI gui(1200,800,video);

  size_t w = video.Streams()[0].Width();
  size_t h = video.Streams()[0].Height();
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

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();
    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    //vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    //cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();

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

