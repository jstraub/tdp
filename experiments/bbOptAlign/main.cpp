/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
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
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>
#include <tdp/preproc/curvature.h>

int main( int argc, char* argv[] )
{
  const std::string inputA = std::string(argv[1]);
  const std::string inputB = std::string(argv[2]);
  const std::string option = (argc > 3) ? std::string(argv[3]) : "";

  bool runOnce = false;
  if (!option.compare("-1")) {
    runOnce = true; 
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
  // use those OpenGL buffers
  
  ManagedHostImage<Vector3fda> vertsA;
  ManagedHostImage<Vector3fda> nsA;
  tdp::LoadPointCloud(inputA, vertsA, nsA);
  ManagedHostImage<Vector3fda> vertsB;
  ManagedHostImage<Vector3fda> nsB;
  tdp::LoadPointCloud(inputB, vertsB, nsB);

  pangolin::GlBuffer vboA, vboB;
  vboA.Reinitialise(pangolin::GlArrayBuffer, vertsA.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  vboA.Upload(vertsA.ptr_, vertsA.SizeBytes(), 0);
  vboB.Reinitialise(pangolin::GlArrayBuffer, vertsB.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  vboB.Upload(vertsB.ptr_, vertsB.SizeBytes(), 0);
//  pangolin::GlBuffer valuebo;
//  valuebo.Reinitialise(pangolin::GlArrayBuffer, vertices.w_,  GL_FLOAT,
//      1, GL_DYNAMIC_DRAW);

  // load and compile shader
  std::string shaderRoot = SHADER_DIR;
  pangolin::GlSlProgram progValueShading;
  progValueShading.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("valueShading.vert"));
  progValueShading.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("valueShading.frag"));
  progValueShading.Link();

  pangolin::Var<bool> showMeanCurvature("ui.show MeanCurv", false, false);
  pangolin::Var<bool> showGausCurvature("ui.show GausCurv", false, false);

  tdp::SE3f T_ab;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (runOnce) break;

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
    // draw the axis

    glColor4f(1.,0.,0.,1.);
    pangolin::glDrawAxis(0.1);
    pangolin::RenderVbo(vboA);

    pangolin::glSetFrameOfReference(T_ab.matrix());
    glColor4f(0.,1.,0.,1.);
    pangolin::glDrawAxis(0.1);
    pangolin::RenderVbo(vboB);
    pangolin::glUnsetFrameOfReference();

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}

