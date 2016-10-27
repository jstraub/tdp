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
#include <tdp/marching_cubes/CIsoSurface.h>
#include <iostream>
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>

int main( int argc, char* argv[] )
{
  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

  bool runOnce = false;
  if (!option.compare("-1")) {
    runOnce = true; 
  }
  std::string meshOutputPath = pangolin::PathParent(input_uri)+std::string("/mesh.ply");
  std::cout << meshOutputPath << std::endl;

  // TODO: use LoadTSDF to also get grid0 and dGrid
  tdp::Vector3fda grid0, dGrid;
  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(0, 0, 0);
  if (!tdp::TSDF::LoadTSDF(input_uri, tsdf, grid0, dGrid)) {
//  if (!tdp::LoadVolume<tdp::TSDFval>(tsdf, input_uri)) {
    pango_print_error("Unable to load volume");
    return 1;
  }
  std::cout << "loaded TSDF volume of size: " << tsdf.w_ << "x" 
    << tsdf.h_ << "x" << tsdf.d_ << std::endl;

//  float test[27] = {
//        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,
//        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,
//        -1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f,-1.0f, 1.0f, 2.0f};

  float xScale = dGrid(0);
  float yScale = dGrid(1);
  float zScale = dGrid(2);
//  float xScale = 6./512;
//  float yScale = 6./512;
//  float zScale = 6./512;

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
  tdp::QuickView viewTsdfSliveView(tsdf.w_,tsdf.h_);
  container.AddDisplay(viewTsdfSliveView);

  // use those OpenGL buffers
  pangolin::GlBuffer vbo;
  pangolin::GlBuffer cbo;
  pangolin::GlBuffer ibo;

  // Add some variables to GUI
  pangolin::Var<float> wThr("ui.weight thr",1,1,100);
  pangolin::Var<float> fThr("ui.tsdf value thr",0.2,0.01,0.5);
  pangolin::Var<bool> recomputeMesh("ui.recompute mesh", true, false);
  pangolin::Var<bool> showTSDFslice("ui.show tsdf slice", false, true);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",tsdf.d_/2,0,tsdf.d_-1);

  // load and compile shader
  std::string shaderRoot = SHADER_DIR;
  pangolin::GlSlProgram colorPc;
  colorPc.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("normalShading.vert"));
  colorPc.AddShaderFromFile(pangolin::GlSlGeometryShader,
      shaderRoot+std::string("normalShading.geom"));
  colorPc.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("normalShading.frag"));
  colorPc.Link();

  tdp::ManagedHostImage<float> tsdfSlice(tsdf.w_, tsdf.h_);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (pangolin::Pushed(recomputeMesh)) {
      // procesing of TSDF
      int64_t start = pangolin::Time_us(pangolin::TimeNow());
      CIsoSurface surface;
      surface.GenerateSurface(&tsdf, 0.0f, xScale, yScale, zScale, wThr, fThr);
      if (!surface.IsSurfaceValid()) {
        pango_print_error("Unable to generate surface");
        return 1;
      }

      int64_t mid = pangolin::Time_us(pangolin::TimeNow());
      size_t nVertices = surface.numVertices();
      size_t nTriangles = surface.numTriangles();

      // TODO: make those tdp::Image<T>
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
      //
      vbo.Reinitialise(pangolin::GlArrayBuffer, nVertices,  GL_FLOAT,         3, GL_DYNAMIC_DRAW);
      cbo.Reinitialise(pangolin::GlArrayBuffer, nVertices,  GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);
      ibo.Reinitialise(pangolin::GlElementArrayBuffer, nTriangles, GL_UNSIGNED_INT,  3, GL_DYNAMIC_DRAW);
      vbo.Upload(vertexStore, sizeof(float) * nVertices * 3, 0);
      cbo.Upload(colorStore,  sizeof(unsigned char) * nVertices * 3, 0);
      ibo.Upload(indexStore,  sizeof(unsigned int) * nTriangles * 3, 0);

      // TODO: dumm to do another copy
      std::vector<float> verts(nVertices * 3);
      verts.assign(vertexStore, vertexStore+nVertices*3);
      std::vector<uint32_t> faces(nTriangles*3);
      faces.assign(indexStore, indexStore+nTriangles*3);

      tinyply::PlyFile plyFile;
      plyFile.add_properties_to_element("vertex", {"x", "y", "z"}, verts);
      plyFile.add_properties_to_element("face", {"vertex_indices"},
          faces, 3, tinyply::PlyProperty::Type::INT8);
      plyFile.comments.push_back("generated from TSDF");

      std::ostringstream outStream;
      plyFile.write(outStream, true);

      std::ofstream out(meshOutputPath);
      out << outStream.str();
      out.close();

      delete [] vertexStore;
      delete [] colorStore;
      delete [] indexStore;

      if (runOnce) break;
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
    // draw the axis
    pangolin::glDrawAxis(0.1);


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
    glDrawElements(GL_TRIANGLES, ibo.num_elements*3, ibo.datatype, 0);
    ibo.Unbind();

    colorPc.Unbind();
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    cbo.Unbind();
    vbo.Unbind();

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    viewTsdfSliveView.Show(showTSDFslice);
    if (viewTsdfSliveView.IsShown()) {
      tdp::Image<tdp::TSDFval> tsdfSliceRaw =
        tsdf.GetImage(std::min((int)tsdf.d_-1,tsdfSliceD.Get()));
      for (size_t i=0; i<tsdfSliceRaw.Area(); ++i) 
        tsdfSlice[i] = tsdfSliceRaw[i].f;
      viewTsdfSliveView.SetImage(tsdfSlice);
    }

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}

