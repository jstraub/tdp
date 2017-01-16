#pragma once

#include <tdp/tsdf/tsdf.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_volume.h>
#include <tdp/eigen/dense.h>
#include <tdp/nn/ann.h>

#include <math.h>
#include <cmath>

namespace tdp {

struct TsdfShapeFields {

  /*
   *  Generate a cylindrical model of the arm. Assume the radius of the arm
   * is on average 1/10 the length of the arm. The model of the arm will be
   * a cylinder with rotational axis on the z axis centered at the origin.
   */
  static float make_cylindrical_point_cloud(Eigen::Matrix<float, 3, Eigen::Dynamic>& points, Eigen::Vector3f& boundingLength, Eigen::Vector3f& center)
  {
    const float PI = 3.1415927f;
    const float HEIGHT_SCALE = 0.75f;
    const float RADIUS = 2 * HEIGHT_SCALE / 10;

    // The dimensions of the box that would completely surround the cylinder
    boundingLength(0) = 0.4f;
    boundingLength(1) = 0.4f;
    boundingLength(2) = 2.0f;

    // The center of the cylinder. Not that this allows the marching cubes
    // reconstruction to place the bounding box at the origin.
    center(0) = 0.2f;
    center(1) = 0.2f;
    center(2) = 1.0f;

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

      points(0, i) = x + center(0);
      points(1, i) = y + center(1);
      points(2, i) = z + center(2);
    }

    return PI * RADIUS * RADIUS * 2 * HEIGHT_SCALE;
  }

  static inline void set_value(tdp::ManagedHostVolume<tdp::TSDFval>&tsdf, int i, int j, int k, float f) {
    tsdf(i, j, k).f = f;
    tsdf(i, j, k).w = 2;
    tsdf(i, j, k).r = 128;
    tsdf(i, j, k).g = 128;
    tsdf(i, j, k).b = 128;
  }

  static inline float outside_cylinder(float x, float y, float z, Eigen::Vector3f& center) {
    const float MAX_Z = 0.75f;
    const float MAX_R = 2 * MAX_Z / 10;

    x -= center(0);
    y -= center(1);
    z -= center(2);

    return z <= -MAX_Z || z >= MAX_Z || sqrt(x * x + y * y) >= MAX_R;
  }

  static void build_tsdf(tdp::ManagedHostVolume<tdp::TSDFval>& tsdf, Eigen::Matrix<float, 3, Eigen::Dynamic>& points, Eigen::Vector3f& scale, Eigen::Vector3f& center)
  {
    // if there are n points in each direction numbered from [0, n-1] that need to map to [-1, 1],
    // then we can map the coordinates by doing (2i / (n - 1) - 1)
    // Note that we need to prevent points inside the surface from being anything but -1

    tdp::Image<tdp::Vector3fda> pc(points.cols(),1,(tdp::Vector3fda*)&(points(0,0)));

    tdp::ANN ann;
    ann.ComputeKDtree(pc);
    Eigen::VectorXi nnIds(1);
    Eigen::VectorXf dists(1);

    float mid_x = (tsdf.w_ - 1) / 2.0f;
    float mid_y = (tsdf.h_ - 1) / 2.0f;
    float mid_z = (tsdf.d_ - 1) / 2.0f;

    for (int i = 0; i < tsdf.w_; i++) {
      float x = scale(0) * (i - mid_x) + center(0);

      for (int j = 0; j < tsdf.h_; j++) {
        float y = scale(1) * (j - mid_y) + center(1);

        for (int k = 0; k < tsdf.d_; k++) {
          float z = scale(2) * (k - mid_z) + center(2);

          float f;
          tdp::Vector3fda q(x, y, z);
          ann.Search(q, 1, 1e-7, nnIds, dists);
          f = sqrt(dists(0));
          if (!outside_cylinder(x, y, z, center)) {
            f *= -1.0f;
          }

          set_value(tsdf, i, j, k, f);
        }
      }
    }
  }
};

}
