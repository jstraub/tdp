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
float make_cylindrical_point_cloud(Eigen::Matrix<float, 3, Eigen::Dynamic>& points)
{
  const float PI = 3.1415927f;
  const float HEIGHT_SCALE = 0.75f;
  const float RADIUS = 2 * HEIGHT_SCALE / 10;

  // Note that the area of the lateral surface of the cylinder to each of the
  // circular faces is 20 : 1 : 1, therefore the first unit of randomness can
  // either determine the height of the point, or if it should be part of the
  // circular faces, it determines the radius from the center of the point
  for (size_t i = 0; i < points.cols(); i++) {
    // Random returns a number from [-1, 1] for each index
    Eigen::Vector2f random = Eigen::Vector2f::Random();
    float x, y, z, theta;

    if (i % 22 == 0) {                          // Top face
      z = HEIGHT_SCALE;
      x = random(0) * RADIUS;
      y = random(1) * RADIUS;

      while (x * x + y * y > RADIUS * RADIUS) {
        random = Eigen::Vector2f::Random();
        x = random(0) * RADIUS;
        y = random(1) * RADIUS;
      }
    } else if (i % 22 == 1) {                   // Bottom face
      z = -HEIGHT_SCALE;
      x = random(0) * RADIUS;
      y = random(1) * RADIUS;

      while (x * x + y * y > RADIUS * RADIUS) {
        random = Eigen::Vector2f::Random();
        x = random(0) * RADIUS;
        y = random(1) * RADIUS;
      }
    } else {                                    // Lateral surface the other 20 times
      z = random(0) * HEIGHT_SCALE;
      theta = random(1) * PI;
      x = RADIUS * cos(theta);
      y = RADIUS * sin(theta);
    }

    points(0, i) = x + 1;
    points(1, i) = y + 1;
    points(2, i) = z + 1;
  }

  return PI * RADIUS * RADIUS * 2 * HEIGHT_SCALE;
}

static inline void setValue(tdp::ManagedHostVolume<tdp::TSDFval>&tsdf, int i, int j, int k, float f) {
  tsdf(i, j, k).f = f;
  tsdf(i, j, k).w = 2;
  tsdf(i, j, k).r = 128;
  tsdf(i, j, k).g = 128;
  tsdf(i, j, k).b = 128;
}

static inline float outside_cylinder(float x, float y, float z, float cx, float cy, float cz) {
  const float MAX_Z = 0.75f;
  const float MAX_R = 2 * MAX_Z / 10;

  x -= cx;
  y -= cy;
  z -= cz;

  return z <= -MAX_Z || z >= MAX_Z || sqrt(x * x + y * y) >= MAX_R;
}

void build_tsdf(tdp::ManagedHostVolume<tdp::TSDFval>& tsdf, Eigen::Matrix<float, 3, Eigen::Dynamic>& points, float xScale, float yScale, float zScale)
{
  // if there are n points in each direction numbered from [0, n-1] that need to map to [-1, 1],
  // then we can map the coordinates by doing (2i / (n - 1) - 1)
  // Note that we need to prevent points inside the surface from being anything but -1

  tdp::Image<tdp::Vector3fda> pc(points.cols(),1,(tdp::Vector3fda*)&(points(0,0)));

  tdp::ANN ann;
  ann.ComputeKDtree(pc);
  Eigen::VectorXi nnIds(1);
  Eigen::VectorXf dists(1);

  for (int i = 0; i < tsdf.w_; i++) {
    float x = xScale * i;
    
    for (int j = 0; j < tsdf.h_; j++) {
      float y = yScale * j;
      
      for (int k = 0; k < tsdf.d_; k++) {
        float z = zScale * k;

        float f;
        tdp::Vector3fda q(x, y, z);
        ann.Search(q, 1, 1e-7, nnIds, dists);
        f = sqrt(dists(0));
        if (!outside_cylinder(x, y, z, 1, 1, 1)) {
          f *= -1.0f;
        }

        setValue(tsdf, i, j, k, f);
      }
    }
  }
}

void build_sphere_tsdf(tdp::ManagedHostVolume<tdp::TSDFval>& tsdf, float r, int n) {
  for (int i = 0; i < tsdf.w_; i++) {
    for (int j = 0; j < tsdf.h_; j++) {
      for (int k = 0; k < tsdf.d_; k++) {
        float x = (i - n / 2);
        float y = (j - n / 2);
        float z = (k - n / 2);

        float f = sqrt(x * x + y * y + z * z) - r;

        setValue(tsdf, i, j, k, f);
      }
    }
  }
}

void calculate_volumes(Eigen::Matrix<float, 3, Eigen::Dynamic>& points,
                      int min, int max, int step, float expectedVolume) {
  std::cout << "Expected: " << expectedVolume << std::endl;
  // cylindrical tsdf
  CIsoSurface surface;

  for (int discretization = min; discretization <= max; discretization += step) {
    tdp::ManagedHostVolume<tdp::TSDFval> tsdf(discretization, discretization, discretization);
    float xScale = 2.0f / (discretization - 1);
    float yScale = 2.0f / (discretization - 1);
    float zScale = 2.0f / (discretization - 1);

    build_tsdf(tsdf, points, xScale, yScale, zScale);
    surface.GenerateSurface(&tsdf, 0.0f, xScale, yScale, zScale, 1.0f, 1.0f);
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
  Eigen::Matrix<float, 3, Eigen::Dynamic> points(3, num_points);
  float volume = make_cylindrical_point_cloud(points);
  std::cout << "Built Point Cloud" << std::endl;

  calculate_volumes(points, 12, 512, 4, volume);
  /*
  int discretization = 12;


  // cylindrical tsdf
  tdp::ManagedHostVolume<tdp::TSDFval> tsdf(discretization, discretization, discretization);
  float xScale = 2.0f / (discretization - 1);
  float yScale = 2.0f / (discretization - 1);
  float zScale = 2.0f / (discretization - 1);

  build_tsdf(tsdf, points, xScale, yScale, zScale);
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
  */
}

