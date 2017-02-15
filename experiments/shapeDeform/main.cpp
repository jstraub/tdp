/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>
#include <random>


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
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/gl_draw.h>

#include <tdp/gui/gui.hpp>
#include <tdp/gui/quickView.h>

#include <tdp/nn/ann.h>
#include <tdp/manifold/S.h>
#include <tdp/manifold/SE3.h>
#include <tdp/data/managed_image.h>

#include <tdp/utils/status.h>
#include <tdp/utils/timer.hpp>
#include <tdp/eigen/std_vector.h>


#include <tdp/laplace_beltrami/laplace_beltrami.h>
/************Declarations***************************************
 ***************************************************************/

/***************************************************************/

/***************************************************************/
//TODO:
//CHECK THE VISUALIZATION
/***************************************************************/

int main( int argc, char* argv[] )
{
//  Test_deform();
//  return 0;
  //Create openGL window - guess sensible dimensions
  int menu_w = 180;
  pangolin::CreateWindowAndBind("GuiBase", 1200+menu_w, 800);
  //Current frame in memory buffer and displayingh
  pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menu_w));
  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //Setup container
  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
      .SetBounds(0.,1.0, pangolin::Attach::Pix(menu_w), 1.0);
  //Define Camera Render Object (for view/scene browsing)
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
    pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
    );
  pangolin::OpenGlRenderState t_cam(
    pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
    pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
    );


  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& view_s = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(s_cam));
  pangolin::View& view_t = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(t_cam));


  container.AddDisplay(view_s);
  container.AddDisplay(view_t);

  // Add variables to pangolin GUI
  pangolin::Var<bool> show("ui.show fMap", true, false);
  pangolin::Var<int> nSamples("ui. nSamples", 500, 100, 10000);
  pangolin::Var<float> max_phi("ui. max phi", M_PI_2, 1e-6, M_PI);


  // use OpenGL buffers
  pangolin::GlBuffer vbo_s, vbo_t;

  tdp::ManagedHostImage<tdp::Vector3fda> pc_s, pc_s_cart, pc_t, pc_t_cart;

  while (!pangolin::ShouldQuit()){
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (pangolin::Pushed(show) || nSamples.GuiChanged() ||
        max_phi.GuiChanged()){
      std::cout << "(Re)running----" << std::endl;
      tdp::GetPointsOnSphere(pc_s, (int)nSamples);
      tdp::toCartisean(pc_s, pc_s_cart);

      tdp::Deform(pc_s, pc_t,(float)max_phi);
      tdp::toCartisean(pc_t, pc_t_cart);

      vbo_s.Reinitialise(pangolin::GlArrayBuffer, pc_s_cart.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_s.Upload(pc_s_cart.ptr_, pc_s_cart.SizeBytes(), 0);
      vbo_t.Reinitialise(pangolin::GlArrayBuffer, pc_t_cart.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_t.Upload(pc_t_cart.ptr_, pc_t_cart.SizeBytes(), 0);
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (view_s.IsShown()) {
      view_s.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(3.);
      glColor4f(1.0f, 0.0f, 0.0f, 0.5f);
      pangolin::RenderVbo(vbo_s);

      glPointSize(5.);
      glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
      pangolin::RenderVbo(vbo_t);
    }

    if (view_t.IsShown()){
        view_t.Activate(t_cam);
        pangolin::glDrawAxis(0.1);

//        glPointSize(4.);
//        glColor3f(0.0f, 0.0f, 1.0f);
//        pangolin::RenderVbo(vbo_t);
    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }

  std::cout << "AY YO!" << std::endl;
  return 0;

}
