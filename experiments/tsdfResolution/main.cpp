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
#include <tdp/gl/shaders.h>

#include <math.h>
#include <cmath>
#include <stdlib.h>

/*
 *  Generate a cylindrical model of the arm. Assume the radius of the arm
 * is on average 1/10 the length of the arm. The model of the arm will be
 * a cylinder with rotational axis on the z axis centered at the origin.
 * The hieght of the cylinder will span from -0.75 to 0.75 units. Hopefully
 * this will somewhat simulate the viewpoint we would have on the arm.
 */
float make_cylindrical_point_cloud(Eigen::Matrix<float, Eigen::Dynamic, 3>& points)
{
  const float PI = 3.1415927f;
  const float HEIGHT_SCALE = 0.75f;
  const float RADIUS = 2 * HEIGHT_SCALE / 10;

  // Note that the area of the lateral surface of the cylinder to each of the
  // circular faces is 20 : 1 : 1, therefore the first unit of randomness can
  // either determine the height of the point, or if it should be part of the
  // circular faces, it determines the radius from the center of the point
  for (size_t i = 0; i < points.rows(); i++) {
    // Random returns a number from [-1, 1] for each index
    Eigen::Vector2f random = Eigen::Vector2f::Random();
    float x, y, z, theta, r;

    if (i % 22 == 0) {                          // Top face
      z = HEIGHT_SCALE;
      r = (random(0) + 1) / 2 * RADIUS;
    } else if (i % 22 == 1) {                   // Bottom face
      z = -HEIGHT_SCALE;
      r = (random(0) + 1) / 2 * RADIUS;
    } else {                                    // Lateral surface the other 20 times
      z = random(0) * HEIGHT_SCALE;
      r = RADIUS;
    }

    theta = random(1) * PI;
    x = RADIUS * cos(theta);
    y = RADIUS * sin(theta);

    points(i,0) = x;
    points(i,1) = y;
    points(i,2) = z;
  }

  return 2 * PI * RADIUS * HEIGHT_SCALE;
}

static inline float outside_cylinder(float x, float y, float z) {
  const float MAX_Z = 0.75f;
  const float MAX_R = 2 * MAX_Z / 10;

  return z <= -MAX_Z || z >= MAX_Z || sqrt(x * x + y * y) >= MAX_R;
}

float min_dist_to_cloud(float x, float y, float z, Eigen::Matrix<float, Eigen::Dynamic, 3>& points) 
{
  float min_dist = 1000000.0f;
  for (int i = 0; i < points.rows(); i++) {
    Eigen::Vector3f v;
    v << x - points(i,0), y - points(i,1), z - points(i,2);
    float dist = v.norm();
    if (dist < min_dist) {
      min_dist = dist;
    }
  }

  return min_dist;
}

void build_tsdf(tdp::ManagedHostVolume<tdp::TSDFval>& tsdf, Eigen::Matrix<float, Eigen::Dynamic, 3>& points)
{
  // if there are n points in each direction numbered from [0, n-1] that need to map to [-1, 1],
  // then we can map the coordinates by doing (2i / (n - 1) - 1)
  // Note that we need to prevent points inside the surface from being anything but -1

  for (int i = 0; i < tsdf.w_; i++) {
    float x = (2.0f * i) / (tsdf.w_ - 1) - 1;
    
    for (int j = 0; j < tsdf.h_; j++) {
      float y = (2.0f * j) / (tsdf.h_ - 1) - 1;
      
      for (int k = 0; k < tsdf.d_; k++) {
        float z = (2.0f * k) / (tsdf.d_ - 1) - 1;

        float f;
        if (outside_cylinder(x, y, z)) {
          f = min_dist_to_cloud(x, y, z, points);
        } else {
          f = -1.0f;
        }

        tsdf(i, j, k).f = f;
        tsdf(i, j, k).w = 2;
        tsdf(i, j, k).r = 128;
        tsdf(i, j, k).g = 128;
        tsdf(i, j, k).b = 128;
      }
    }
  }
}

int main( int argc, char* argv[] )
{
  bool runOnce = false;

  // Generate the same point cloud each time
  srand(0);
  int num_points = 22 * 100;
  Eigen::Matrix<float, Eigen::Dynamic, 3> points(num_points, 3);
  float volume = make_cylindrical_point_cloud(points);

  std::cout << "Built Point Cloud" << std::endl;

  int resolution = 20;

  tdp::SE3f T_wG;
  tdp::Vector3fda grid0, dGrid;
  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(resolution, resolution, resolution);

  build_tsdf(tsdf, points);

  float xScale = 6./512;
  float yScale = 6./512;
  float zScale = 6./512;

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

