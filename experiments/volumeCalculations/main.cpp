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
#include <tdp/nn/ann.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/tsdf/tsdf.h>
#include <tdp/data/managed_volume.h>
#include <tdp/marching_cubes/CIsoSurface.h>
#include <iostream>
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>

#include <tdp/tsdf/tsdf_shapes.h>

#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <tdp/reconstruction/plane.h>
#include <tdp/reconstruction/volumeReconstruction.h>
#include "test.h"

void render_plane(tdp::Reconstruction::Plane plane,
                  pangolin::Var<bool>& flip_normal,
                  pangolin::Var<bool>& show_normal,
                  pangolin::GlBuffer& vbo,
                  pangolin::GlBuffer& ibo,
                  auto& shader,
                  Eigen::Vector3f boundingLength,
                  Eigen::Vector3f center) {
  // TODO: Find a better way to do this. Possibly w/ plane intersecting bounding box
  float minX = center(0) - boundingLength(0) / 2;
  float maxX = center(0) + boundingLength(0) / 2;
  float minY = center(1) - boundingLength(1) / 2;
  float maxY = center(1) + boundingLength(1) / 2;

  float vertexStore[4 * 3];
  vertexStore[0 * 3 + 0] = minX;
  vertexStore[0 * 3 + 1] = minY;
  vertexStore[0 * 3 + 2] = plane.find_z_coordinate(minX, minY);
  vertexStore[1 * 3 + 0] = minX;
  vertexStore[1 * 3 + 1] = maxY;
  vertexStore[1 * 3 + 2] = plane.find_z_coordinate(minX, maxY);
  vertexStore[2 * 3 + 0] = maxX;
  vertexStore[2 * 3 + 1] = maxY;
  vertexStore[2 * 3 + 2] = plane.find_z_coordinate(maxX, maxY);
  vertexStore[3 * 3 + 0] = maxX;
  vertexStore[3 * 3 + 1] = minY;
  vertexStore[3 * 3 + 2] = plane.find_z_coordinate(maxX, minY);

  unsigned int indexStore[2 * 3];
  if (flip_normal) {
    indexStore[0 * 3 + 0] = 0;
    indexStore[0 * 3 + 1] = 1;
    indexStore[0 * 3 + 2] = 3;
    indexStore[1 * 3 + 0] = 3;
    indexStore[1 * 3 + 1] = 1;
    indexStore[1 * 3 + 2] = 2;
  } else {
    indexStore[0 * 3 + 0] = 0;
    indexStore[0 * 3 + 1] = 3;
    indexStore[0 * 3 + 2] = 1;
    indexStore[1 * 3 + 0] = 1;
    indexStore[1 * 3 + 1] = 3;
    indexStore[1 * 3 + 2] = 2;
  }

  vbo.Reinitialise(pangolin::GlArrayBuffer, 4,  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  ibo.Reinitialise(pangolin::GlElementArrayBuffer, 2, GL_UNSIGNED_INT,  3, GL_DYNAMIC_DRAW);
  vbo.Upload(vertexStore, sizeof(float) * 4 * 3, 0);
  ibo.Upload(indexStore,  sizeof(unsigned int) * 2 * 3, 0);

  if (vbo.IsValid() && ibo.IsValid()) {
    vbo.Bind();
    glVertexPointer(vbo.count_per_element, vbo.datatype, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    shader.Bind();
    ibo.Bind();
    glDrawElements(GL_TRIANGLES,ibo.num_elements * 3, ibo.datatype, 0);
    ibo.Unbind();
    shader.Unbind();

    glDisableClientState(GL_VERTEX_ARRAY);
    vbo.Unbind();

    if (show_normal) {
      // TODO: Figure out how to render an arrow
    }
  }
}

int main( int argc, char* argv[] )
{
  runtests();
  bool runOnce = false;

  // Generate the same point cloud each time
  srand(0);
  int num_points = 22 * 1000;

  Eigen::Vector3f boundingLength;
  Eigen::Vector3f center;
  Eigen::Matrix<float, 3, Eigen::Dynamic> points(3, num_points);

  float volume = tdp::TsdfShapeFields::make_cylindrical_point_cloud(points, boundingLength, center);
  std::cout << "Built Point Cloud" << std::endl;
  std::cout << "Bounding Length: " << boundingLength.transpose() << std::endl;
  std::cout << "Center: " << center.transpose() << std::endl;
  std::cout << "Volume: " << volume << std::endl;

  int discretization = 128;
  Eigen::Vector3f scale = boundingLength / (discretization - 1);

  // cylindrical tsdf
  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(discretization, discretization, discretization);
  float xScale = scale(0);
  float yScale = scale(1);
  float zScale = scale(2);

  tdp::TsdfShapeFields::build_tsdf(tsdf, points, scale, center);
  std::cout << "Finished building TSDF" << std::endl;

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

  pangolin::GlBuffer vbo_pc;
  vbo_pc.Reinitialise(pangolin::GlArrayBuffer, num_points,  GL_FLOAT,         3, GL_DYNAMIC_DRAW);
  vbo_pc.Upload((float *)&(points(0,0)), sizeof(float) * num_points * 3, 0);

  // Add some variables to GUI
  pangolin::Var<float> wThr("ui.weight thr",1,1,100);
  pangolin::Var<float> fThr("ui.tsdf value thr",1.,0.01,0.5);
  pangolin::Var<bool> recomputeMesh("ui.recompute mesh", true, false);
  pangolin::Var<bool> showTSDFslice("ui.show tsdf slice", false, true);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",tsdf.d_/2,0,tsdf.d_-1);

  pangolin::Var<bool> showPointCloud("ui.show point cloud", false, true);

  // TODO: Use theta and phi angles to determine orientation of the plane. the radius is then "d"
  //       This can then let us put bounds min and max on the possibile orientations
  float maxd = sqrt(boundingLength[0] * boundingLength[0] + boundingLength[1] * boundingLength[1] + boundingLength[2] * boundingLength[2]);
  // Plane 1 cutoffs
  pangolin::Var<float> pl1_nx("ui.plane_1 nx", 0, 0, boundingLength[0]);
  pangolin::Var<float> pl1_ny("ui.plane_1 ny", 0, 0, boundingLength[1]);
  pangolin::Var<float> pl1_nz("ui.plane_1 nz", 1, 0, boundingLength[2]);
  pangolin::Var<float> pl1_d("ui.plane_1 d", maxd / 3,              0, maxd);
  pangolin::Var<bool>  pl1_flip_normal("ui.plane_1 flip normal", true, true);
  pangolin::Var<bool>  pl1_show_normal("ui.plane_1 show normal", false, true);
  pangolin::GlBuffer   pl1_vbo;
  pangolin::GlBuffer   pl1_ibo;

  // Plane 2 cutoffs
  pangolin::Var<float> pl2_nx("ui.plane_2 nx", 0,  0, boundingLength[0]);
  pangolin::Var<float> pl2_ny("ui.plane_2 ny", 0,  0, boundingLength[1]);
  pangolin::Var<float> pl2_nz("ui.plane_2 nz", 1,  0, boundingLength[2]);
  pangolin::Var<float> pl2_d("ui.plane_2 d", maxd * 2 / 3,               0, maxd);
  pangolin::Var<bool>  pl2_flip_normal("ui.plane_2 flip normal", false, true);
  pangolin::Var<bool>  pl2_show_normal("ui.plane_2 show normal", false, true);
  pangolin::GlBuffer   pl2_vbo;
  pangolin::GlBuffer   pl2_ibo;

  pangolin::Var<bool> recomputeVolume("ui.recompute volume", true, false);

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
      surface.getColors(colorStore);

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

      delete [] vertexStore;
      delete [] colorStore;
      delete [] indexStore;

      if (runOnce) break;
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    // draw the axis
    pangolin::glDrawAxis(0.1);

    // Render the Marching Cubes Mesh
    // pangolin::RenderVboIboCbo(vbo, ibo, cbo, true, true);
    if (vbo.IsValid() && cbo.IsValid() && ibo.IsValid()) {
      vbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
      cbo.Bind();
      glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);

      pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
      pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
      auto& shader = tdp::Shaders::Instance()->normalMeshShader_;
      shader.Bind();
      shader.SetUniform("P",P);
      shader.SetUniform("MV",MV);

      ibo.Bind();
      glDrawElements(GL_TRIANGLES, ibo.num_elements*3,
          ibo.datatype, 0);
      ibo.Unbind();

      shader.Unbind();
      glDisableVertexAttribArray(1);
      glDisableVertexAttribArray(0);
      cbo.Unbind();
      vbo.Unbind();
    }

    // Draw point cloud if desired
    if (showPointCloud) {
      glColor3f(1,0,0);
      pangolin::RenderVbo(vbo_pc);
    }

    // Draw Intersecting Planes
    auto& shader = tdp::Shaders::Instance()->normalMeshShader_;
    int sign = pl1_flip_normal ? -1 : 1;
    tdp::Reconstruction::Plane pl1(sign * pl1_nx, sign * pl1_ny, sign * pl1_nz, sign * pl1_d);

    sign = pl2_flip_normal ? -1 : 1;
    tdp::Reconstruction::Plane pl2(sign * pl2_nx, sign * pl2_ny, sign * pl2_nz, sign * pl2_d);

    render_plane(pl1, pl1_flip_normal, pl1_show_normal, pl1_vbo, pl1_ibo, shader, boundingLength, center);
    render_plane(pl2, pl2_flip_normal, pl2_show_normal, pl2_vbo, pl2_ibo, shader, boundingLength, center);

    if (pangolin::Pushed(recomputeVolume)) {
      std::cout << "Estimated volume: " << tdp::Reconstruction::volume_in_bounds(tsdf, pl1, pl2, scale) << std::endl;
    }

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
