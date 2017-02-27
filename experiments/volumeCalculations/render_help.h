#pragma once

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

#include <vector>

void render_point_cloud(
     pangolin::GlBuffer& vbo,
     std::vector<float>& points,
     float r, float g, float b
) {
  vbo.Reinitialise(pangolin::GlArrayBuffer, points.size() / 3,  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vbo.Upload(points.data(), sizeof(float) * points.size(), 0);

  glColor3f(r, g, b);
  pangolin::RenderVbo(vbo);
}

void render_surface_normals(const float* vertices,
                            const size_t numVertices,
                            const uint32_t* indices,
                            const size_t numTriangles) {
    glColor3f(1,0,0);
    glBegin(GL_LINES);
    for (size_t i = 0; i < numTriangles; i++) {
      size_t c1 = indices[3 * i + 0],
             c2 = indices[3 * i + 1],
             c3 = indices[3 * i + 2];
      tdp::Vector3fda v1(vertices[3 * c1 + 0], vertices[3 * c1 + 1], vertices[3 * c1 + 2]);
      tdp::Vector3fda v2(vertices[3 * c2 + 0], vertices[3 * c2 + 1], vertices[3 * c2 + 2]);
      tdp::Vector3fda v3(vertices[3 * c3 + 0], vertices[3 * c3 + 1], vertices[3 * c3 + 2]);
      tdp::Vector3fda centroid = (v1 + v2 + v3) / 3;

      tdp::Vector3fda normal = (v2 - v1).cross(v3 - v1).normalized() / 100;

      tdp::Vector3fda endpoint = centroid + normal;
      glVertex3f(centroid(0), centroid(1), centroid(2));
      glVertex3f(endpoint(0), endpoint(1), endpoint(2));
    }
    glEnd();
}

void render_bounding_box_corners(
                  pangolin::GlBuffer& vbo,
                  const tdp::Vector3fda& corner1,
                  const tdp::Vector3fda& corner2) {
  size_t numVertices = 8;
  float x[2] = {corner1(0), corner2(0)};
  float y[2] = {corner1(1), corner2(1)};
  float z[2] = {corner1(2), corner2(2)};
  float vertexStore[numVertices * 3];

  size_t vertex = 0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++, vertex++) {
        vertexStore[vertex * 3 + 0] = x[i];
        vertexStore[vertex * 3 + 1] = y[j];
        vertexStore[vertex * 3 + 2] = z[k];
      }

  vbo.Reinitialise(pangolin::GlArrayBuffer, numVertices,  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vbo.Upload(vertexStore, sizeof(float) * numVertices * 3, 0);

  if (vbo.IsValid()) {
    glColor3f(0, 1, 0);
    pangolin::RenderVbo(vbo);
  }
}

void render_plane(const tdp::Reconstruction::Plane& plane,
                  pangolin::GlBuffer& vbo,
                  pangolin::GlBuffer& ibo,
                  auto& shader,
                  const tdp::Vector3fda& corner1,
                  const tdp::Vector3fda& corner2) {
  size_t numVertices = 6;
  size_t numTriangles = 4;

  float vertexStore[numVertices * 3];
  unsigned int indexStore[numTriangles * 3];

  // find the intersecting polygon between the plane and the bounding box of the TSDF
  tdp::Vector3fda polygon[6];
  tdp::Reconstruction::get_vertices_of_intersection(polygon, plane, corner1, corner2);

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
    // make the plane blue
    glColor3f(0,0,1);
    tdp::RenderVboIbo(vbo, ibo);
    // Render a line pointing in the direction of the normal as well
    // Make it red to contrast with the plane
    // Keep in mind that the scale size is in terms of meters, so decrease unit normal
    tdp::Vector3fda other_endpoint = polygon[0] + 0.1 * plane.unit_normal();
    glColor3f(1,0,0);
    glBegin(GL_LINES);
       glVertex3f(polygon[0](0), polygon[0](1), polygon[0](2));
       glVertex3f(other_endpoint(0), other_endpoint(1), other_endpoint(2));
    glEnd();
  }
}

