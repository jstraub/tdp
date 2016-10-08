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
  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

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
  
  std::vector<float> verts;
  std::vector<uint32_t> tris;
  std::ifstream in(input_uri);
  tinyply::PlyFile ply(in);

  for (auto e : ply.get_elements()) {
    std::cout << "element - " << e.name << " (" << e.size << ")" 
      << std::endl;
    for (auto p : e.properties) {
      std::cout << "\tproperty - " << p.name << " (" 
        << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
    }
  }
  std::cout << std::endl;
  ply.request_properties_from_element("vertex", {"x", "y", "z"}, verts);
  ply.request_properties_from_element("face", {"vertex_indices"}, tris);
  ply.read(in);
  std::cout << "loaded ply file: "
    << verts.size() << " " << tris.size() << std::endl;

  std::vector<uint8_t> cols(verts.size(), 128);
  tdp::Image<tdp::Vector3fda> vertices(verts.size()/3,1,(tdp::Vector3fda*)&verts[0]);
  tdp::Image<tdp::Vector3uda> tri(tris.size()/3,1,(tdp::Vector3uda*)&tris[0]);
  tdp::Image<tdp::Vector3bda> color(cols.size()/3,1,(tdp::Vector3bda*)&cols[0]);

  tdp::ManagedHostImage<tdp::Vector3fda> n(vertices.w_,1);
  tdp::ManagedHostImage<tdp::Vector3fda> meanCurv(vertices.w_,1);
  tdp::ManagedHostImage<float> gausCurv(vertices.w_,1);
  std::map<uint32_t,std::vector<uint32_t>> neigh;
  std::cout << "Compute neighborhood" << std::endl;
  tdp::ComputeNeighborhood(vertices, tri, n, neigh);
  std::cout << "Compute curvature" << std::endl;
  tdp::ComputeCurvature(vertices, tri, neigh, meanCurv, gausCurv);

  pangolin::GlBuffer vbo;
  pangolin::GlBuffer valuebo;
  pangolin::GlBuffer cbo;
  pangolin::GlBuffer ibo;
  vbo.Reinitialise(pangolin::GlArrayBuffer, vertices.w_,  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  valuebo.Reinitialise(pangolin::GlArrayBuffer, vertices.w_,  GL_FLOAT,
      1, GL_DYNAMIC_DRAW);
  cbo.Reinitialise(pangolin::GlArrayBuffer, vertices.w_,
      GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
  ibo.Reinitialise(pangolin::GlElementArrayBuffer, tri.w_,
      GL_UNSIGNED_INT,  3, GL_DYNAMIC_DRAW);

  vbo.Upload(vertices.ptr_, vertices.SizeBytes(), 0);
  cbo.Upload(color.ptr_,  color.SizeBytes(), 0);
  ibo.Upload(tri.ptr_,  tri.SizeBytes(), 0);

  // load and compile shader
  std::string shaderRoot = SHADER_DIR;
  pangolin::GlSlProgram progNormalShading;
  progNormalShading.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("normalShading.vert"));
  progNormalShading.AddShaderFromFile(pangolin::GlSlGeometryShader,
      shaderRoot+std::string("normalShading.geom"));
  progNormalShading.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("normalShading.frag"));
  progNormalShading.Link();

  pangolin::GlSlProgram progValueShading;
  progValueShading.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("valueShading.vert"));
  progValueShading.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("valueShading.frag"));
  progValueShading.Link();

  pangolin::Var<bool> showMeanCurvature("ui.show MeanCurv", false, false);
  pangolin::Var<bool> showGausCurvature("ui.show GausCurv", false, false);

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
    pangolin::glDrawAxis(0.1);

    if (showGausCurvature || showMeanCurvature) {
      progValueShading.Bind();
      progValueShading.SetUniform("P",P);
      progValueShading.SetUniform("MV",MV);
      std::pair<double,double> minMax;
      if (showGausCurvature) {
        minMax = gausCurv.MinMax();
        valuebo.Upload(gausCurv.ptr_,  gausCurv.SizeBytes(), 0);
      } else if (showMeanCurvature) {
        minMax = meanCurv.MinMax();
        valuebo.Upload(meanCurv.ptr_,  meanCurv.SizeBytes(), 0);
      }
      progValueShading.SetUniform("minValue", float(minMax.first));
      progValueShading.SetUniform("maxValue", float(minMax.second));

      valuebo.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0); 
    } else {
      progNormalShading.Bind();
      progNormalShading.SetUniform("P",P);
      progNormalShading.SetUniform("MV",MV);

      cbo.Bind();
      glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0); 
    }
    vbo.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 

    glEnableVertexAttribArray(0);                                               
    glEnableVertexAttribArray(1);                                               

    ibo.Bind();
    glDrawElements(GL_TRIANGLES, ibo.num_elements*3, ibo.datatype, 0);
    ibo.Unbind();

    if (showGausCurvature || showMeanCurvature) {
      progValueShading.Unbind();
      glDisableVertexAttribArray(1);
      valuebo.Unbind();
    } else {
      progNormalShading.Unbind();
      glDisableVertexAttribArray(1);
      cbo.Unbind();
    }
    glDisableVertexAttribArray(0);
    vbo.Unbind();

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}

