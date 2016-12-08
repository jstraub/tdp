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

// Returns true if a voxel is completely inside the surface
bool inside_surface(tdp::ManagedHostVolume<tdp::TSDFval>& tsdf, size_t x, size_t y, size_t z) {
  bool inside = true;

  inside &= tsdf(x    , y    , z    ).f <= 0;
  inside &= tsdf(x + 1, y    , z    ).f <= 0;
  inside &= tsdf(x    , y + 1, z    ).f <= 0;
  inside &= tsdf(x    , y    , z + 1).f <= 0;
  inside &= tsdf(x + 1, y + 1, z    ).f <= 0;
  inside &= tsdf(x + 1, y    , z + 1).f <= 0;
  inside &= tsdf(x    , y + 1, z + 1).f <= 0;
  inside &= tsdf(x + 1, y + 1, z + 1).f <= 0;

  return inside;
}

// Returns true if a voxel is completely outside of a plane of intersection
bool outside_plane(Eigen::Vector3f normal, float d, size_t i, size_t j, size_t k, Eigen::Vector3f scale) {
  bool less = false, greater = false;
  for (int dx = 0; dx <= 1; dx++)
    for (int dy = 0; dy <= 1; dy++)
      for(int dz = 0; dz <= 1; dz++) {
        Eigen::Vector3f x((i + dx) * scale(0), (j + dy) * scale(1), (k + dz) * scale(2));

        // Calculate the distance to the plane from each corner
        float out = normal.dot(x) - d;

        // Ignore equality because the volume won't be affected
        // if only one corner intersects the plane
        less    |= out < 0;
        greater |= out > 0;
      }
  return !greater;
}

// Returns true if the voxel intersections the plane
bool intersects(Eigen::Vector3f normal, float d, size_t i, size_t j, size_t k, Eigen::Vector3f scale) {
  bool less = false, greater = false;

  for (int dx = 0; dx <= 1; dx++)
    for (int dy = 0; dy <= 1; dy++)
      for(int dz = 0; dz <= 1; dz++) {
        Eigen::Vector3f x((i + dx) * scale(0), (j + dy) * scale(1), (k + dz) * scale(2));

        // Calculate the distance to the plane from each corner
        float out = normal.dot(x) - d;

        // Ignore equality because the volume won't be affected
        // if only one corner intersects the plane
        less    |= out < 0;
        greater |= out > 0;
      }

  return less && greater;
}

// first dimension specifies the index of the corner to be denoted v0
// the second dimension lists the mapping from (v0 - v7) -> (0 - 7)
// i.e. ordered_index_from_index[i][j] gives the index of vj given that v0 = i
const int ordered_index_from_index[8][8] = {
  {0,1,3,4,5,2,7,6},
  {1,2,0,5,6,3,4,7},
  {2,3,1,6,7,0,5,4},
  {3,0,2,7,4,1,6,5},
  {4,5,0,7,6,1,3,2},
  {5,6,1,4,7,2,0,3},
  {6,7,2,5,4,3,1,0},
  {7,4,3,6,5,0,2,1}
};

int find_v0(Eigen::Vector3f normal, float d, Eigen::Vector3f* tmp) {

  // Note that if d is negative, then we could flip the signs of the normal, and d to make it positive
  // the maximization assumes postive d
  int index = 0;
  int sign = (0.0f < d) - (d < 0.0f);
  float maxVal = sign * (normal.dot(tmp[0]) - d);

  for (int i = 1; i < 8; i++) {
    Eigen::Vector3f p = tmp[i];

    float val = sign * (normal.dot(p) - d);

    if (val > maxVal) {
      maxVal = val;
      index = i;
    }
  }

  return index;
}

/*
 * Returns the volume >= to the plane
 */
float percentVolumeRight(Eigen::Vector3f normal, float d, size_t i, size_t j, size_t k, Eigen::Vector3f scale) {
  // If we let the plane with the given normal sweep from d = inifinity downwards, let v0 be defined as
  // the first vertex it would intersect, v7 be the last vertex it would intersect, and let all other vertices
  // be numbered according to the right hand rule

  // further let us number the vertices of the cube from 0 - 7 as follows
  // (i    , j    , k    ) -> 0
  // (i + 1, j    , k    ) -> 1
  // (i + 1, j + 1, k    ) -> 2
  // (i    , j + 1, k    ) -> 3
  // (i    ,      , k + 1) -> 4
  // (i + 1, j    , k + 1) -> 5
  // (i + 1, j + 1, k + 1) -> 6
  // (i    , j + 1, k + 1) -> 7

  // then we just need to figure out which vertex is "first" and then from that we have a deterministic
  // mapping from numbers (0-7) -> (v0 - v7).


  Eigen::Vector3f tmp[8] = {
    Eigen::Vector3f((i    ) * scale(0), (j    ) * scale(1), (k    ) * scale(2)),
    Eigen::Vector3f((i + 1) * scale(0), (j    ) * scale(1), (k    ) * scale(2)),
    Eigen::Vector3f((i + 1) * scale(0), (j + 1) * scale(1), (k    ) * scale(2)),
    Eigen::Vector3f((i    ) * scale(0), (j + 1) * scale(1), (k    ) * scale(2)),
    Eigen::Vector3f((i    ) * scale(0), (j    ) * scale(1), (k + 1) * scale(2)),
    Eigen::Vector3f((i + 1) * scale(0), (j    ) * scale(1), (k + 1) * scale(2)),
    Eigen::Vector3f((i + 1) * scale(0), (j + 1) * scale(1), (k + 1) * scale(2)),
    Eigen::Vector3f((i    ) * scale(0), (j + 1) * scale(1), (k + 1) * scale(2))
  };

  int index = find_v0(normal, d, tmp);
  Eigen::Vector3f v[8];
  for (int t = 0; t < 8; t++) {
    v[t] = tmp[ordered_index_from_index[index][t]];
  }

  // Now given v0 - v7, we can calculate for the exact vertices of the intersections in an order that would
  // define a polygon. There are at most 6 vertices that arise from the intersection of a plane and a
  // rectangular prism. If there needs to be less vertices, then we will simply duplicate vertices to
  // create a degenerate side of length 0.

  // P0: Intersection on E0->1, E1->4, E4->7
  // P1: Intersection on E1->5 or P0
  // P2: Intersection on E0->2, E2->5, E5->7
  // P3: Intersection on E2->6 or P2
  // P4: Intersection on E0->3, E3->6, E6->7
  // P5: Intersection on E3->4 or P4

  Eigen::Vector3f p[6];
  Eigen::Vector3f E01 = tmp[1] - tmp[0];
  Eigen::Vector3f E14 = tmp[4] - tmp[1];
  Eigen::Vector3f E47 = tmp[7] - tmp[4];
  Eigen::Vector3f E15 = tmp[5] - tmp[1];
  Eigen::Vector3f E02 = tmp[2] - tmp[0];
  Eigen::Vector3f E25 = tmp[5] - tmp[2];
  Eigen::Vector3f E57 = tmp[7] - tmp[5];
  Eigen::Vector3f E26 = tmp[6] - tmp[2];
  Eigen::Vector3f E03 = tmp[3] - tmp[0];
  Eigen::Vector3f E36 = tmp[6] - tmp[3];
  Eigen::Vector3f E67 = tmp[7] - tmp[6];
  Eigen::Vector3f E34 = tmp[4] - tmp[3];

  float lambda;


  // Given the set of vertices, we can now compute the volume bounded by the polygon and rectangular prism.
  // Note that the volume we are interested in is the volume that includes the point v0

  return 0.0f;
}

/*
 * Returns the volume <= to the plane
 */
float percentVolumeLeft(Eigen::Vector3f normal, float d, size_t i, size_t j, size_t k, Eigen::Vector3f scale) {
  return 1 - percentVolumeRight(normal, d, i, j, k, scale);
}

/*
  left and right should be the coefficients for the hessian normal form of the plane n dot x = d.
  assume the indices are such that 0 -> x, 1 -> y, 2 -> z, 3 -> d
  scale should be the x, y, z sidelength values
  Assumes that the normal of the left and right planes point towards each other. e.g. n_l dot n_r < 0
 */
float volume_exclude_non_surface_voxels(
        tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
        Eigen::Vector3f n_left,
        float d_left,
        Eigen::Vector3f n_right,
        float d_right,
        Eigen::Vector3f scale
) {
  // Cases:
  //   Surface Voxel -> ignore
  //   Interior voxel ->
  //        Inside bounds?    -> add
  //        Intersect bounds? -> calculate fraction and add
  //   Exterior voxels -> ignore
  // Sources of error
  //    * lack of surface voxel volume (hollow cylinder of volume)
  //    * if we add surface voxels that are not on the intersecting plane then we miss 2 rings of voxels

  float volume = 0.0;

  for (size_t k = 0; k < tsdf.d_ - 1; k++)
    for (size_t j = 0; j < tsdf.h_ - 1; j++)
      for (size_t i = 0; i < tsdf.w_ - 1; i++) {
        if (!inside_surface(tsdf, i, j, k))
          continue;

        if (outside_plane(n_left, d_left, i, j, k, scale) ||
            outside_plane(n_right, d_right, i, j, k, scale))
          continue;

        float percentVolume;

        if (intersects(n_left, d_left, i, j, k, scale)) {
          percentVolume = percentVolumeRight(n_left, d_left, i, j, k, scale);
        } else if (intersects(n_right, d_right, i, j, k, scale)) {
          percentVolume = percentVolumeLeft(n_right, d_right, i, j, k, scale);
        } else {
          percentVolume = 1.0f;
        }
        volume += scale[0] * scale[1] * scale[2] * percentVolume;
    }


    return volume;
}

void calculate_volumes(Eigen::Matrix<float, 3, Eigen::Dynamic>& points,
                      float *boundingLength, float* center,
                      int min, int max, int step, float expectedVolume) {
  std::cout << "Expected: " << expectedVolume << std::endl;
  // cylindrical tsdf
  CIsoSurface surface;

  for (int discretization = min; discretization <= max; discretization += step) {
    tdp::ManagedHostVolume<tdp::TSDFval> tsdf(discretization, discretization, discretization);
    float scale[3] = {
      boundingLength[0] / (discretization - 1),
      boundingLength[1] / (discretization - 1),
      boundingLength[2] / (discretization - 1)
    };

    tdp::TsdfShapeFields::build_tsdf(tsdf, points, scale, center);
    surface.GenerateSurface(&tsdf, 0.0f, scale[0], scale[1], scale[2], 1.0f, 1.0f);
    if (!surface.IsSurfaceValid()) {
      pango_print_error("Unable to generate surface");
    }

    std::cout << discretization << " " << surface.getVolume() << std::endl;
  }
}

int main( int argc, char* argv[] )
{
  bool runOnce = false;

  // Generate the same point cloud each time
  srand(0);
  int num_points = 22 * 1000;
  float boundingLength  [3];
  float center          [3];
  Eigen::Matrix<float, 3, Eigen::Dynamic> points(3, num_points);
  float volume = tdp::TsdfShapeFields::make_cylindrical_point_cloud(points, boundingLength, center);
  std::cout << "Built Point Cloud" << std::endl;

  //calculate_volumes(points, boundingLength, center, 128, 128, 4, volume);
  
  int discretization = 128;
  float scale[3] = {
    boundingLength[0] / (discretization - 1),
    boundingLength[1] / (discretization - 1),
    boundingLength[2] / (discretization - 1)
  };

  // cylindrical tsdf
  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(discretization, discretization, discretization);
  float xScale = scale[0];
  float yScale = scale[1];
  float zScale = scale[2];

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

  // load and compile shader
  //std::string shaderRoot = SHADER_DIR;
  //pangolin::GlSlProgram colorPc;
  //colorPc.AddShaderFromFile(pangolin::GlSlVertexShader, 
  //    shaderRoot+std::string("normalShading.vert"));
  //colorPc.AddShaderFromFile(pangolin::GlSlGeometryShader,
  //    shaderRoot+std::string("normalShading.geom"));
  //colorPc.AddShaderFromFile(pangolin::GlSlFragmentShader,
  //    shaderRoot+std::string("normalShading.frag"));
  //colorPc.Link();

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

      int64_t end = pangolin::Time_us(pangolin::TimeNow());
      std::cout << "GenerateSurface time: " << (mid - start) / 1e6 << std::endl;
      std::cout << "copy time: " << (end - mid) / 1e6 << std::endl;
      std::cout << "Number of Vertices: " << nVertices << std::endl;
      std::cout << "Number of Triangles: " << nTriangles << std::endl;
      std::cout << "Volume: " << surface.getVolume() << std::endl;
      std::cout << "Expect: " << volume << std::endl;
      std::cout << wThr << " " << fThr << std::endl;
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

//    pangolin::RenderVboIboCbo(vbo, ibo, cbo, true, true);
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

      glColor3f(1,0,0);
      pangolin::RenderVbo(vbo_pc);
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

