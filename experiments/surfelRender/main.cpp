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
#include <tdp/gl/render.h>
#include <tdp/gl/gl_draw.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/normals.h>

#include <tdp/gui/gui.hpp>
#include <tdp/io/tinyply.h>

int main( int argc, char* argv[] )
{
  std::string pathToPc = "";
  if( argc > 1 ) {
    pathToPc = std::string(argv[1]);
  }

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



  uint32_t w=640;
  uint32_t h=480;
  tdp::QuickView viewRender(w,h);
  container.AddDisplay(viewRender);
//    .SetHandler(new pangolin::Handler3D(s_cam2));
  
  uint32_t N = 9;
  // host image: image in CPU memory
  tdp::ManagedHostImage<tdp::Vector3fda> pc(N);
  tdp::ManagedHostImage<tdp::Vector3fda> n(N);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(N);
  tdp::ManagedHostImage<float> r(N);
  rgb.Fill(tdp::Vector3bda(255,0,0));
  n.Fill(tdp::Vector3fda(0.,0.,-1.));

  if (pathToPc.size() > 0) {
    LoadPointCloud(pathToPc, pc, n, rgb, true);
    N = pc.Area();
//    rgb.Reinitialise(N,1);
//    rgb.Fill(tdp::Vector3bda(255,0,0));
    r.Reinitialise(N,1);
  }

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,N,GL_FLOAT,3);
  pangolin::GlBuffer nbo(pangolin::GlArrayBuffer,N,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,N,GL_UNSIGNED_BYTE,3);
  pangolin::GlBuffer rbo(pangolin::GlArrayBuffer,N,GL_FLOAT,1);

  pangolin::GlTexture tex(w, h, GL_RGB);
  pangolin::GlRenderBuffer render(w, h, GL_DEPTH_COMPONENT);
  pangolin::GlFramebuffer fbo(tex, render);

//  tdp::ManagedHostImage<tdp::Vector4bda> rgbaI(w,h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgbI(w,h);

  // Add some variables to GUI
  pangolin::Var<float> radius("ui.radius",0.1,0.001,0.1);
  pangolin::Var<float> scale("ui.scale",1.,0.,3.);
  pangolin::Var<float> dx("ui.dx",0.5,0.001,1.);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    r.Fill(radius);
    uint32_t j=0;
    for (int32_t u=-1; u<2; u++)
      for (int32_t v=-1; v<2; v++) {
        pc[j++] = tdp::Vector3fda(u*dx, v*dx, 1.);
      }

    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    rbo.Upload(r.ptr_,r.SizeBytes(), 0);
    nbo.Upload(n.ptr_,n.SizeBytes(), 0);
    // Draw 3D stuff

    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();

    pangolin::glDrawAxis(0.1);
    tdp::RenderSurfels( vbo, nbo, cbo, rbo, 4., P, MV);
    glColor3f(0,1,0);
    for (size_t i=0; i<N; ++i) {
      tdp::glDrawLine(pc[i], pc[i]+scale*radius*n[i]);
    }
    glDisable(GL_DEPTH_TEST);

//    viewRender.Activate(s_cam2);
//    P = s_cam2.GetProjectionMatrix();
//    MV = s_cam2.GetModelViewMatrix();


    fbo.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glPointSize(1);
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    tdp::RenderSurfels( vbo, nbo, cbo, rbo, 10., P, MV);
    glColor3f(0,1,0);
    for (size_t i=0; i<N; ++i) {
      tdp::glDrawLine(pc[i], pc[i]+scale*radius*n[i]);
    }
    fbo.Unbind();
    glPopAttrib();
    glFinish();
    tex.Download(rgbI.ptr_, GL_RGB, GL_UNSIGNED_BYTE);

    viewRender.Activate();
    viewRender.FlipTextureY()=true;
////    viewRender.UpdateView();
//    viewRender.glSetViewOrtho();
//    tex.Bind();
//    viewRender.glRenderTexture(tex);
//    tex.Unbind();
    viewRender.SetImage(rgbI);
//    viewRender.UpdateView();

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}


