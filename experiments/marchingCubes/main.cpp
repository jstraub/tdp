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

#include <tdp/tsdf/tsdf.h>
#include <tdp/data/managed_volume.h>
#include "CIsoSurface.h"
#include <iostream>
#include <pangolin/utils/timer.h>

int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  const std::string input_uri = std::string(argv[1]);
  const std::string output_uri = (argc > 2) ? std::string(argv[2]) : dflt_output_uri;

  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(0, 0, 0);
  if (!tdp::LoadVolume<tdp::TSDFval>(tsdf, input_uri)) {
    pango_print_error("Unable to load volume");
    return 1;
  }

//  float test[27] = {
//        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,
//        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,
//        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f};

  float xScale = 0.002f;
  float yScale = 0.002f;
  float zScale = 0.002f;
  size_t xDim = tsdf.w_;
  size_t yDim = tsdf.h_;
  size_t zDim = tsdf.d_;
  float *points = new float[xDim * yDim * zDim];
  for (size_t i = 0; i < xDim * yDim * zDim; i++) {
      points[i] = tsdf.ptr_[i].f;
  }

  // procesing of TSDF
  int64_t start = pangolin::Time_us(pangolin::TimeNow());
  CIsoSurface<float> surface;
  surface.GenerateSurface(points, 0.0f, xDim - 1, yDim - 1, zDim - 1, xScale, yScale, zScale);
  if (!surface.IsSurfaceValid()) {
    pango_print_error("Unable to generate surface");
    return 1;
  }

  int64_t mid = pangolin::Time_us(pangolin::TimeNow());
  size_t nVertices = surface.numVertices();
  size_t nTriangles = surface.numTriangles();

  float* vertexStore = new float[nVertices * 3];
  unsigned char* colorStore = new unsigned char[nVertices * 3];
  unsigned int* indexStore = new unsigned int[nTriangles * 3];

  surface.getVertices(vertexStore);
  surface.getIndices(indexStore);
  for (size_t i = 0; i < nVertices * 3; i++) {
    colorStore[i] = 128;
  }
  int64_t end = pangolin::Time_us(pangolin::TimeNow());

  std::cout << "GenerateSurface time: " << (mid - start) / 1e6 << std::endl;
  std::cout << "copy time: " << (end - mid) / 1e6 << std::endl;

  std::cout << "Number of Vertices: " << nVertices << std::endl;
  std::cout << "Number of Triangles: " << nTriangles << std::endl;
//  for (size_t i = 0; i < 4; i++) {
//      std::cout << vertexStore[3 * i] << " ";
//      std::cout << vertexStore[3 * i + 1] << " ";
//      std::cout << vertexStore[3 * i + 2] << std::endl;
//  }
//  for (size_t i = 0; i < 4; i++) {
//      std::cout << indexStore[3 * i] << " ";
//      std::cout << indexStore[3 * i + 1] << " ";
//      std::cout << indexStore[3 * i + 2] << std::endl;
//  }

  // Create OpenGL window - guess sensible dimensions
  int menue_w = 180;
  pangolin::CreateWindowAndBind( "GuiBase", 1200+menue_w, 800);
  // current frame in memory buffer and displaying.
  pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menue_w));
  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // setup container
  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
    .SetBounds(0., 1.0, pangolin::Attach::Pix(menue_w), 1.0);
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(d_cam);

  // use those OpenGL buffers
  pangolin::GlBuffer vbo;
  pangolin::GlBuffer cbo;
  pangolin::GlBuffer ibo;

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);


  pangolin::GlSlProgram colorPc;
  colorPc.AddShaderFromFile(pangolin::GlSlVertexShader,
      "/home/jstraub/workspace/tdp/shaders/normalShading.vert");
  colorPc.AddShaderFromFile(pangolin::GlSlGeometryShader,
      "/home/jstraub/workspace/tdp/shaders/normalShading.geom");
  colorPc.AddShaderFromFile(pangolin::GlSlFragmentShader,
      "/home/jstraub/workspace/tdp/shaders/normalShading.frag");
  colorPc.Link();

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
    // draw the axis
    pangolin::glDrawAxis(0.1);

    vbo.Reinitialise(pangolin::GlArrayBuffer, nVertices,  GL_FLOAT,         3, GL_DYNAMIC_DRAW);
    cbo.Reinitialise(pangolin::GlArrayBuffer, nVertices,  GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
    ibo.Reinitialise(pangolin::GlElementArrayBuffer, nTriangles, GL_UNSIGNED_INT,  3, GL_DYNAMIC_DRAW);
    vbo.Upload(vertexStore, sizeof(float) * nVertices * 3, 0);
    cbo.Upload(colorStore,  sizeof(unsigned char) * nVertices * 3, 0);
    ibo.Upload(indexStore,  sizeof(unsigned int) * nTriangles * 3, 0);

    vbo.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
    cbo.Bind();
    glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0); 

    glEnableVertexAttribArray(0);                                               
    glEnableVertexAttribArray(1);                                               

    colorPc.Bind();
    colorPc.SetUniform("P",P);
    colorPc.SetUniform("MV",MV);

    ibo.Bind();
    glDrawElements(GL_TRIANGLES,ibo.num_elements, ibo.datatype, 0);
    ibo.Unbind();

    colorPc.Unbind();
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    cbo.Unbind();
    vbo.Unbind();

    // render point cloud
    //cbo.Bind();
    //glColorPointer(cbo.count_per_element, cbo.datatype, 0, 0);
    //glEnableClientState(GL_COLOR_ARRAY);

    //vbo.Bind();
    //glVertexPointer(vbo.count_per_element, vbo.datatype, 0, 0);
    //glEnableClientState(GL_VERTEX_ARRAY);

    //ibo.Bind();
    //glDrawElements(GL_TRIANGLES,ibo.num_elements, ibo.datatype, 0);
    //ibo.Unbind();

    //glDisableClientState(GL_VERTEX_ARRAY);
    //vbo.Unbind();

    //glDisableClientState(GL_COLOR_ARRAY);
    //cbo.Unbind();

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}

