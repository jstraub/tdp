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



int main( int argc, char* argv[] ){
  //Test_simplePc();
  //return 1;
  tdp::ManagedHostImage<tdp::Vector3fda> pc_s(1000,1);
  tdp::ManagedHostImage<tdp::Vector3fda> ns_s(1000,1);

  tdp::ManagedHostImage<tdp::Vector3fda> pc_t(pc_s.Area(),1);
  tdp::ManagedHostImage<tdp::Vector3fda> ns_t(ns_s.Area(),1);

  const std::string input_uri = std::string(argv[1]);
  std::ifstream in(input_uri);


  std::cout << "input pc: " << input_uri << std::endl;

  std::vector<float> verts;
  tinyply::PlyFile ply(in);
  for (auto e : ply.get_elements()){
    std::cout << "element - " << e.name << " (" << e.size << ")"
              << std::endl;
    for (auto p : e.properties){
      std::cout << "\tproperty - " << p.name << "("
                << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
    }
  }
  std::cout << std::endl;
  ply.request_properties_from_element("vertex", {"x", "y", "z"}, verts);
  ply.read(in);
  std::cout << "loaded ply file: "
            << verts.size() << std::endl;

  tdp::Image<tdp::Vector3fda> vertices(verts.size()/3, 1, (tdp::Vector3fda*)&verts[0]);
  std::cout << "number of vertices: " << vertices.Area() << std::endl;


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
  pangolin::View& viewPc = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewPc);

  // Add variables to pangolin GUI
  pangolin::Var<bool> showFMap("ui.show fMap", true, false);

  // use those OpenGL buffers
  pangolin::GlBuffer vbo;
  //-- upload point cloud positions
  vbo.Reinitialise(pangolin::GlArrayBuffer, vertices.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vbo.Upload(vertices.ptr_, vertices.SizeBytes(), 0);


  // Stream and display video
  while(!pangolin::ShouldQuit()){
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (viewPc.IsShown()) {
      viewPc.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(1.);
      glColor3f(0.0f, 1.0f, 0.0f);
      pangolin::RenderVbo(vbo);

    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }

  return 0;
}
