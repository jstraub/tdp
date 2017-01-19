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
#include <tdp/marching_cubes/marching_cubes.h>
#include <iostream>
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/render.h>

#include <tdp/tsdf/tsdf_shapes.h>

#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <tdp/reconstruction/plane.h>
#include <tdp/reconstruction/volumeReconstruction.h>
#include "test.h"

void render_plane(tdp::Reconstruction::Plane plane,
                  pangolin::Var<bool>& flip_normal,
                  pangolin::GlBuffer& vbo,
                  pangolin::GlBuffer& ibo,
                  auto& shader,
                  tdp::Vector3fda corner1,
                  tdp::Vector3fda corner2) {
  size_t numVertices = 6;
  size_t numTriangles = 4;

  float vertexStore[numVertices * 3];
  unsigned int indexStore[numTriangles * 3];

  // find the intersecting polygon between the plane and the bounding box of the TSDF
  tdp::Vector3fda polygon[6];
  if (flip_normal) {
    tdp::Reconstruction::get_vertices_of_intersection(polygon, plane.flip(), corner1, corner2);
  } else {
    tdp::Reconstruction::get_vertices_of_intersection(polygon, plane.flip(), corner1, corner2);
  }

  // copy the data into the buffers
  for (int i = 0; i < numVertices; i++) {
    vertexStore[i * 3 + 0] = polygon[i](0);
    vertexStore[i * 3 + 1] = polygon[i](1);
    vertexStore[i * 3 + 2] = polygon[i](2);
  }

  for (int i = 0; i < numTriangles; i++) {
    indexStore[i * 3 + 0] = 0;
    indexStore[i * 3 + 1] = i + 1;
    indexStore[i * 3 + 2] = i + 2;
  }

  // Load the data into the opengl buffers
  vbo.Reinitialise(pangolin::GlArrayBuffer, numVertices,  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  ibo.Reinitialise(pangolin::GlElementArrayBuffer, numTriangles, GL_UNSIGNED_INT,  3, GL_DYNAMIC_DRAW);
  vbo.Upload(vertexStore, sizeof(float) * numVertices * 3, 0);
  ibo.Upload(indexStore,  sizeof(unsigned int) * numTriangles * 3, 0);

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
  }
}

int main( int argc, char* argv[] )
{
  runtests();

  bool runOnce = false;

  // Generate the same point cloud each time
  // srand(0);
  // int num_points = 22 * 1000;

  // tdp::Vector3fda boundingLength;
  // tdp::Vector3fda center;
  // Eigen::Matrix<float, 3, Eigen::Dynamic> points(3, num_points);

  // float volume = tdp::TsdfShapeFields::make_cylindrical_point_cloud(points, boundingLength, center);
  // std::cout << "Built Point Cloud" << std::endl;
  // std::cout << "Bounding Length: " << boundingLength.transpose() << std::endl;
  // std::cout << "Center: " << center.transpose() << std::endl;
  // std::cout << "Volume: " << volume << std::endl;

  // int discretization = 128;
  // tdp::Vector3fda scale = boundingLength / (discretization - 1);

  // // cylindrical tsdf
  // tdp::ManagedHostVolume<tdp::TSDFval> tsdf(discretization, discretization, discretization);
  // float xScale = scale(0);
  // float yScale = scale(1);
  // float zScale = scale(2);

  // tdp::TsdfShapeFields::build_tsdf(tsdf, points, scale, center);

  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

  if (!option.compare("-1")) {
    runOnce = true;
  }

  tdp::SE3f T_wG;
  tdp::Vector3fda grid0, dGrid;
  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(0, 0, 0);
  if (!tdp::TSDF::LoadTSDF(input_uri, tsdf, T_wG, grid0, dGrid)) {
    pango_print_error("Unable to load volume");
    return 1;
  }
  std::cout << "loaded TSDF volume of size: " << tsdf.w_ << "x"
    << tsdf.h_ << "x" << tsdf.d_ << std::endl
    << T_wG << std::endl;

  // Define opposite corners properly scaled to real world coordinates
  tdp::Vector3fda corner1(0, 0, 0);
  tdp::Vector3fda corner2((tsdf.w_ - 1) * dGrid(0), (tsdf.h_ - 1) * dGrid(1), (tsdf.d_ - 1) * dGrid(2));

  // Finish converting to real space
  corner1 = T_wG * (corner1 + grid0);
  corner2 = T_wG * (corner2 + grid0);
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
  pangolin::GlBuffer meshVbo;
  pangolin::GlBuffer meshCbo;
  pangolin::GlBuffer meshIbo;

  // pangolin::GlBuffer vbo_pc;
  // vbo_pc.Reinitialise(pangolin::GlArrayBuffer, num_points,  GL_FLOAT,         3, GL_DYNAMIC_DRAW);
  // vbo_pc.Upload((float *)&(points(0,0)), sizeof(float) * num_points * 3, 0);

  // Add some variables to GUI
  pangolin::Var<float> marchCubeswThr("ui.weight thr",1,1,100);
  pangolin::Var<float> marchCubesfThr("ui.tsdf value thr",1.,0.01,0.5);
  pangolin::Var<bool> recomputeMesh("ui.recompute mesh", true, false);
  pangolin::Var<bool> showTSDFslice("ui.show tsdf slice", false, true);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",tsdf.d_/2,0,tsdf.d_-1);

  pangolin::Var<bool> showPointCloud("ui.show point cloud", false, true);

  // TODO: Use theta and phi angles to determine orientation of the plane. the radius is then "d"
  //       This can then let us put bounds min and max on the possibile orientations
  float minX = std::min(corner1(0), corner2(0)),
        minY = std::min(corner1(1), corner2(1)),
        minZ = std::min(corner1(2), corner2(2));
  float maxX = std::max(corner1(0), corner2(0)),
        maxY = std::max(corner1(1), corner2(1)),
        maxZ = std::max(corner1(2), corner2(2));
  float maxD = tdp::Vector3fda(maxX - minX, maxY - minY, maxZ - minZ).norm();

  // Plane 1 cutoffs
  pangolin::Var<float> pl1_nx("ui.plane_1 nx", 0, minX, maxX);
  pangolin::Var<float> pl1_ny("ui.plane_1 ny", 0, minY, maxY);
  pangolin::Var<float> pl1_nz("ui.plane_1 nz", 1, minZ, maxZ);
  pangolin::Var<float> pl1_d("ui.plane_1 d",   (maxZ - minZ) / 2,    0, maxD);
  pangolin::Var<bool>  pl1_flip_normal("ui.plane_1 flip normal", true, true);
  pangolin::GlBuffer   pl1_vbo;
  pangolin::GlBuffer   pl1_ibo;

  // Plane 2 cutoffs
  pangolin::Var<float> pl2_nx("ui.plane_2 nx", 0, minX, maxX);
  pangolin::Var<float> pl2_ny("ui.plane_2 ny", 0, minY, maxY);
  pangolin::Var<float> pl2_nz("ui.plane_2 nz", 1, minZ, maxZ);
  pangolin::Var<float> pl2_d("ui.plane_2 d",   (maxZ - minZ) / 2,    0, maxD);
  pangolin::Var<bool>  pl2_flip_normal("ui.plane_2 flip normal", false, true);
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
      tdp::ComputeMesh(tsdf, grid0, dGrid,
          T_wG, meshVbo, meshCbo, meshIbo, marchCubeswThr, marchCubesfThr);
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    // draw the axis
    pangolin::glDrawAxis(0.1);

    // Render the Marching Cubes Mesh
    // pangolin::RenderVboIboCbo(vbo, ibo, cbo, true, true);
    tdp::RenderVboIboCbo(meshVbo, meshIbo, meshCbo);

    // Draw point cloud if desired
    // if (showPointCloud) {
    //   glColor3f(1,0,0);
    //   pangolin::RenderVbo(vbo_pc);
    // }

    // Draw Intersecting Planes
    auto& shader = tdp::Shaders::Instance()->normalMeshShader_;
    int sign = pl1_flip_normal ? -1 : 1;
    tdp::Reconstruction::Plane pl1(sign * pl1_nx, sign * pl1_ny, sign * pl1_nz, sign * pl1_d);

    sign = pl2_flip_normal ? -1 : 1;
    tdp::Reconstruction::Plane pl2(sign * pl2_nx, sign * pl2_ny, sign * pl2_nz, sign * pl2_d);

    render_plane(pl1, pl1_flip_normal, pl1_vbo, pl1_ibo, shader, corner1, corner2);
    render_plane(pl2, pl2_flip_normal, pl2_vbo, pl2_ibo, shader, corner1, corner2);

    if (pangolin::Pushed(recomputeVolume)) {
      std::cout << "Estimated volume: "
                << tdp::Reconstruction::volume_in_bounds_with_voxel_counting(tsdf, pl1, pl2, grid0, dGrid, T_wG)
                << std::endl;
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
