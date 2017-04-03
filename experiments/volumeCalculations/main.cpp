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
#include <tdp/marching_cubes/marching_cubes.h>
#include <iostream>
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/render.h>

#include <tdp/nn_cuda/nn_cuda.h>

#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <functional>
#include <tdp/reconstruction/plane.h>
#include <tdp/reconstruction/volumeReconstruction.h>
#include <tdp/filters/tsdfFilters.h>
#include "test.h"
#include "render_help.h"
#include "filters.h"
#include <vector>

inline std::function<void(size_t, size_t, size_t)> makeVectorFill(
       std::vector<float>& vec,
       const tdp::Vector3fda& grid0,
       const tdp::Vector3fda& dGrid,
       const tdp::SE3f& T_wG
) {
  vec.clear();

  return [&](size_t i, size_t j, size_t k) {
    tdp::Vector3fda point(i * dGrid(0), j * dGrid(1), k * dGrid(2));
    point = T_wG * (point + grid0);
    vec.push_back(point(0));
    vec.push_back(point(1));
    vec.push_back(point(2));
  };
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
  tdp::ManagedHostVolume<tdp::TSDFval> tsdfBase(0, 0, 0);
  if (!tdp::TSDF::LoadTSDF(input_uri, tsdf, T_wG, grid0, dGrid)) {
    pango_print_error("Unable to load volume");
    return 1;
  }
  tdp::TSDF::LoadTSDF(input_uri, tsdfBase, T_wG, grid0, dGrid);
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTsdfBuf1(tsdf.w_, tsdf.h_, tsdf.d_);
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTsdfBuf2(tsdf.w_, tsdf.h_, tsdf.d_);

  std::cout << "loaded TSDF volume of size: " << tsdf.w_ << "x"
    << tsdf.h_ << "x" << tsdf.d_ << std::endl
    << T_wG << std::endl;
  std::cout << "0, 0, 0: " << grid0.transpose() << std::endl;
  std::cout << "255, 255, 255: "
            << (T_wG * ((tdp::Vector3fda(256 * dGrid(0), 256 * dGrid(1), 256 * dGrid(2))) + grid0)).transpose()
            << std::endl;
  std::cout << "Scale: " << dGrid.transpose() << std::endl;

  // Define opposite corners properly scaled to real world coordinates
  tdp::Vector3fda corner1(0, 0, 0);
  tdp::Vector3fda corner2((tsdf.w_ - 1) * dGrid(0), (tsdf.h_ - 1) * dGrid(1), (tsdf.d_ - 1) * dGrid(2));

  // Finish converting to real space
  corner1 = T_wG * (corner1 + grid0);
  corner2 = T_wG * (corner2 + grid0);

  std::cout << "Corner 1: " << corner1 << std::endl;
  std::cout << "Corner 2: " << corner2 << std::endl;

  std::cout << "Finished building TSDF" << std::endl;

  // Create OpenGL window - guess sensible dimensions
  int menue_w = 180;
  pangolin::CreateWindowAndBind( "GuiBase", 1200+menue_w, 800);
  std::cout << " Create window" << std::endl;
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
  pangolin::Var<float> marchCubeswThr("ui.weight thr",1,0,100);
  pangolin::Var<float> marchCubesfThr("ui.tsdf value thr",1.,0.01,2);
  pangolin::Var<bool> recomputeMesh("ui.recompute mesh", true, false);
  pangolin::Var<bool> showTSDFslice("ui.show tsdf slice", false, true);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",tsdf.d_/2,0,tsdf.d_-1);

  pangolin::Var<bool> showPointCloud("ui.show point cloud", false, true);
  pangolin::Var<bool> reset("ui.reset", false, false);

  // Using cartesian coordinates instead of spherical because the TSDF is not
  // centered at the origin. So it is a tad difficult to manually move planes
  // around to properly cut the arm in spherical coordinates.
  float minX = std::min(corner1(0), corner2(0)),
        minY = std::min(corner1(1), corner2(1)),
        minZ = std::min(corner1(2), corner2(2));
  float maxX = std::max(corner1(0), corner2(0)),
        maxY = std::max(corner1(1), corner2(1)),
        maxZ = std::max(corner1(2), corner2(2));

  float farX = std::max(std::abs(minX), std::abs(maxX));
  float farY = std::max(std::abs(minY), std::abs(maxY));
  float farZ = std::max(std::abs(minZ), std::abs(maxZ));
  float closeX = std::min(std::abs(minX), std::abs(maxX));
  float closeY = std::min(std::abs(minY), std::abs(maxY));
  float closeZ = std::min(std::abs(minZ), std::abs(maxZ));
  float maxD = tdp::Vector3fda(farX, farY, farZ).norm();
  float minD = tdp::Vector3fda(closeX, closeY, closeZ).norm();

  // Plane 1 cutoffs
  //pangolin::Var<float> pl1_nx("ui.plane_1 nx", 1, -1, 1);
  //pangolin::Var<float> pl1_ny("ui.plane_1 ny", 0, -1, 1);
  //pangolin::Var<float> pl1_nz("ui.plane_1 nz", 0, -1, 1);
  //pangolin::Var<float> pl1_d("ui.plane_1 d",   -maxD / 3,    -maxD, maxD);
  pangolin::Var<float> pl1_nx("ui.plane_1 nx", 1, -1, 1);
  pangolin::Var<float> pl1_ny("ui.plane_1 ny", -.02381f, -1, 1);
  pangolin::Var<float> pl1_nz("ui.plane_1 nz", 0.01193f, -1, 1);
  pangolin::Var<float> pl1_d("ui.plane_1 d",   -0.1380f,    -maxD, maxD);
  pangolin::Var<bool>  pl1_flip_normal("ui.plane_1 flip normal", true, true);
  pangolin::GlBuffer   pl1_vbo;
  pangolin::GlBuffer   pl1_ibo;

  // Plane 2 cutoffs
  // pangolin::Var<float> pl2_nx("ui.plane_2 nx", 1, -1, 1);
  // pangolin::Var<float> pl2_ny("ui.plane_2 ny", 0, -1, 1);
  // pangolin::Var<float> pl2_nz("ui.plane_2 nz", 0, -1, 1);
  // pangolin::Var<float> pl2_d("ui.plane_2 d",   maxD / 3,    -maxD, maxD);
  pangolin::Var<float> pl2_nx("ui.plane_2 nx", 1, -1, 1);
  pangolin::Var<float> pl2_ny("ui.plane_2 ny", -.02381f, -1, 1);
  pangolin::Var<float> pl2_nz("ui.plane_2 nz", 0.01193f, -1, 1);
  pangolin::Var<float> pl2_d("ui.plane_2 d",   0.0700f,    -maxD, maxD);
  pangolin::Var<bool>  pl2_flip_normal("ui.plane_2 flip normal", false, true);
  pangolin::GlBuffer   pl2_vbo;
  pangolin::GlBuffer   pl2_ibo;

  // Saving raw data for filtering analysis
  size_t numVertices = 0;
  size_t numTriangles = 0;
  float* vertexStore = new float[numVertices * 3];
  uint32_t* indexStore = new uint32_t[numTriangles * 3];

  pangolin::Var<bool> recomputeVolume("ui.recompute volume", true, false);

  tdp::ManagedHostImage<float> tsdfSlice(tsdf.w_, tsdf.h_);

  bool first = true;
  pangolin::Var<bool> render_bounding_box("ui.show bounding box", false, true);
  pangolin::GlBuffer boundingBoxVbo;

  pangolin::Var<bool> render_normals("ui.render surface normals", false, true);

  // Variables used for filtering points during volume calculations
  tdp::ManagedHostImage<tdp::Vector3fda> centroids;
  tdp::ManagedHostImage<tdp::Vector3fda> normals;
  tdp::NN_Cuda nn;
  Eigen::VectorXi nnIds(1);
  Eigen::VectorXf dists(1);

  // Filters and point clouds for the filtered points
  pangolin::Var<bool> applyPlanes("ui.apply planes", false, false);
  //pangolin::Var<bool> filterUninitialized("ui.filter uninit", false, false);
  //pangolin::Var<bool> showFiltered("ui.show filtered uninit", false, true);
  pangolin::Var<bool> applyMedianFilter("ui.apply median filter", false, false);
  pangolin::Var<bool> fillEdges("ui.fill edges", false, false);
  pangolin::Var<bool> showFilled("ui.show filled", false, true);
  pangolin::Var<bool> filterPositive("ui.filter positive", false, false);
  pangolin::Var<bool> showPositiveFilter("ui.show pos", false, true);
  pangolin::Var<bool> filterNegative("ui.filter negative", false, false);
  pangolin::Var<bool> showNegativeFilter("ui.show neg", false, true);
  pangolin::Var<bool> showVolumeCloud("ui.show volume", false, true);
  pangolin::Var<bool> showTSDFSlice("ui.show tsdf slice plane", false, true);

  pangolin::GlBuffer smallCluster_vbo;
  pangolin::GlBuffer edgeFill_vbo;
  pangolin::GlBuffer f_pos_vbo;
  pangolin::GlBuffer f_neg_vbo;
  pangolin::GlBuffer volume_vbo;
  pangolin::GlBuffer slice_vbo;

  std::vector<float> smallCluster_points;
  std::vector<float> edgeFill_points;
  std::vector<float> f_pos_points;
  std::vector<float> f_neg_points;
  std::vector<float> volume_points;
  std::vector<float> slice_points;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (pangolin::Pushed(recomputeMesh)) {
      tdp::ComputeMesh(tsdf, grid0, dGrid,
          T_wG, meshVbo, meshCbo, meshIbo, marchCubeswThr, marchCubesfThr);
      delete[] vertexStore;
      delete[] indexStore;
      numVertices = meshVbo.num_elements;
      numTriangles = meshIbo.num_elements;
      vertexStore = new float[numVertices * 3];
      indexStore = new uint32_t[numTriangles * 3];
      meshVbo.Download(vertexStore, sizeof(float) * numVertices * 3, 0);
      meshIbo.Download(indexStore, sizeof(uint32_t) * numTriangles * 3, 0);
      set_up_nn(centroids, normals, nn, vertexStore, numVertices, indexStore, numTriangles);

      tdp::Image<tdp::Vector3fda> verts(numVertices, 1, (tdp::Vector3fda*) vertexStore);
      tdp::Image<tdp::Vector3uda> faces(numTriangles, 1, (tdp::Vector3uda*) indexStore);

      std::vector<std::string> comments;
      comments.push_back("saved after multiple filters");
      std::string meshOutputPath = pangolin::PathParent(input_uri) + std::string("/mesh.ply");
      std::cout << "Mesh Path: " << meshOutputPath << std::endl;
      SaveMesh(meshOutputPath, verts, faces, true, comments);
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    // draw the axis
    pangolin::glDrawAxis(0.1);

    // draw bounding box
    if (render_bounding_box) {
      render_bounding_box_corners(boundingBoxVbo, corner1, corner2);
    }

    if (pangolin::Pushed(reset)) {
      tsdf.CopyFrom(tsdfBase);
    }

    // if (pangolin::Pushed(filterUninitialized)) {
    //   auto func = makeVectorFill(smallCluster_points, grid0, dGrid, T_wG);
    //   filterBlackRegions(tsdf, func);
    // }

    // if (showFiltered) {
    //   render_point_cloud(smallCluster_vbo, smallCluster_points, 1, 0, 0);
    // }

    if (pangolin::Pushed(applyMedianFilter)) {
      cuTsdfBuf1.CopyFrom(tsdf);
      // Necessary so that the RGB values are preserved. Maybe we should
      // copy RGB as well as depth and weight from the tsdf points of the medians? hm...
      cuTsdfBuf2.CopyFrom(tsdf);
      tdp::TSDFFilters::medianFilter(cuTsdfBuf1, cuTsdfBuf2);
      tsdf.CopyFrom(cuTsdfBuf2);
    }

    if (pangolin::Pushed(fillEdges)) {
      auto func = makeVectorFill(edgeFill_points, grid0, dGrid, T_wG);
      fillInFromEdges(tsdf, func);
    }

    if (showFilled) {
      render_point_cloud(edgeFill_vbo, edgeFill_points, 0, 0, 1);
    }

    if (pangolin::Pushed(filterPositive)) {
      auto func = makeVectorFill(f_pos_points, grid0, dGrid, T_wG);
      filterPositiveRegions(tsdf, func);
    }

    if (showPositiveFilter) {
      render_point_cloud(f_pos_vbo, f_pos_points, 0, 0, 1);
    }

    if (pangolin::Pushed(filterNegative)) {
      auto func = makeVectorFill(f_neg_points, grid0, dGrid, T_wG);
      filterNegativeRegions(tsdf, func);
    }

    if (showNegativeFilter) {
      render_point_cloud(f_neg_vbo, f_neg_points, 0, 0, 1);
    }

    if (showVolumeCloud) {
      render_point_cloud(volume_vbo, volume_points, 0, 0, 1);
    }

    if (showTSDFSlice) {
      render_point_cloud(slice_vbo, slice_points, 0, 1, 0);
    }

    // Render the Marching Cubes Mesh
    // pangolin::RenderVboIboCbo(vbo, ibo, cbo, true, true);
    tdp::RenderVboIboCbo(meshVbo, meshIbo, meshCbo);

    // Render the surface normals to make sure the triangles are properly oriented
    if (render_normals) {
      render_surface_normals(vertexStore, numVertices, indexStore, numTriangles);
    }

    // Draw point cloud if desired
    // if (showPointCloud) {
    //   glColor3f(1,0,0);
    //   pangolin::RenderVbo(vbo_pc);
    // }

    // Draw Intersecting Planes if in frame
    auto& shader = tdp::Shaders::Instance()->normalMeshShader_;
    int sign = pl1_flip_normal ? -1 : 1;
    tdp::Reconstruction::Plane pl1(sign * pl1_nx, sign * pl1_ny, sign * pl1_nz, sign * pl1_d);

    sign = pl2_flip_normal ? -1 : 1;
    tdp::Reconstruction::Plane pl2(sign * pl2_nx, sign * pl2_ny, sign * pl2_nz, sign * pl2_d);

    if (tdp::Reconstruction::intersect_type(pl1, corner1, corner2) ==
        tdp::Reconstruction::IntersectionType::INTERSECTS) {
      render_plane(pl1, pl1_vbo, pl1_ibo, shader, corner1, corner2);
    }
    if (tdp::Reconstruction::intersect_type(pl2, corner1, corner2) ==
        tdp::Reconstruction::IntersectionType::INTERSECTS) {
      render_plane(pl2, pl2_vbo, pl2_ibo, shader, corner1, corner2);
    }

    if (pangolin::Pushed(recomputeVolume) && !first) {
      // auto func = make_inside_surface_filter(centroids, normals, nn, nnIds, dists);
      auto func = [&](tdp::Vector3fda point) {
        volume_points.push_back(point(0));
        volume_points.push_back(point(1));
        volume_points.push_back(point(2));
        return true;
      };
      std::cout << "Estimated volume: "
                << tdp::Reconstruction::volume_in_bounds_with_voxel_counting(
                            tsdf, pl1, pl2, grid0, dGrid, T_wG,
                            func)
                << std::endl;
    }

    first = false;

    glDisable(GL_DEPTH_TEST);

    if (pangolin::Pushed(applyPlanes)) {
      cuTsdfBuf1.CopyFrom(tsdf);
      tdp::TSDFFilters::applyCuttingPlanes(cuTsdfBuf1, pl1, pl2, grid0, dGrid, T_wG);
      tsdf.CopyFrom(cuTsdfBuf1);
      std::cout << "Distance Between Planes: " << pl1.distance_to_parallel_plane(pl2.flip()) << std::endl;
    }

    // Draw 2D stuff
    viewTsdfSliveView.Show(showTSDFslice);
    if (viewTsdfSliveView.IsShown()) {
      int d = std::min((int)tsdf.d_-1, tsdfSliceD.Get());
      tdp::Image<tdp::TSDFval> tsdfSliceRaw = tsdf.GetImage(d);
      for (size_t i=0; i<tsdfSliceRaw.Area(); ++i)
        tsdfSlice[i] = tsdfSliceRaw[i].f;
      viewTsdfSliveView.SetImage(tsdfSlice);

      auto func = makeVectorFill(slice_points, grid0, dGrid, T_wG);
      for (size_t i = 0; i < tsdf.w_; i++)
        for (size_t j = 0; j < tsdf.h_; j++)
          func(i, j, d);
    }

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  delete[] vertexStore;
  delete[] indexStore;
  return 0;
}
