#include <iostream>

#include <tdp/reconstruction/volumeReconstruction.h>

namespace tdp {
namespace Reconstruction {

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

// TODO: Slow by a factor of 8 (each point calculated 8 times)
IntersectionType intersect_type(Plane plane, size_t i, size_t j, size_t k, Eigen::Vector3f scale) {
  bool hasInside = false, hasOutside = false;

  for (int dx = 0; dx <= 1; dx++)
    for (int dy = 0; dy <= 1; dy++)
      for(int dz = 0; dz <= 1; dz++) {
        Eigen::Vector3f x((i + dx) * scale(0), (j + dy) * scale(1), (k + dz) * scale(2));

        // Calculate the distance to the plane from each corner
        float out = plane.distance_to(x);

        // Non negative distance implies that the vertex is on the side of the plane that would
        // be included in the volume
        hasOutside |= out > 0;
        hasInside  |= out <= 0;
      }

  if (hasInside && hasOutside) {
    return IntersectionType::INTERSECTS;
  } else if (hasInside && !hasOutside) {
    return IntersectionType::INSIDE;
  } else if (!hasInside && hasOutside) {
    return IntersectionType::OUTSIDE;
  } else {
    std::cerr << "Error: Found impossible point that is not inside, not outside, and doesn't intersect a plane" << std::endl;
    exit(1);
  }
}

int find_v0(Plane plane, Eigen::Vector3f* tmp) {

  // Note that if d is negative, then we could flip the signs of the normal, and d to make it positive
  // the maximization assumes postive d
  int index = 0;
  float maxVal = plane.distance_to(tmp[0]);

  for (int i = 1; i < 8; i++) {
    float val = plane.distance_to(tmp[i]);

    if (val > maxVal) {
      maxVal = val;
      index = i;
    }
  }

  return index;
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

/*
 * Returns the percentage of the volume of the voxel that is on the "positive" side of the
 * plane as defined by the normal of the plane
 */
float percent_volume(Plane plane, size_t i, size_t j, size_t k, Eigen::Vector3f scale) {
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

  int index = find_v0(plane, tmp);
  Eigen::Vector3f v[8];
  for (int t = 0; t < 8; t++) {
    v[t] = tmp[ordered_index_from_index[index][t]];
  }

  // Lets also store 6 lists for the set of vertices on each face,
  // allowing us to reconstruct triangles on said faces as is necessary
  // TODO: Finish this


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
  // TODO: Finish this
  float numerator = -1; 

  // Given the set of vertices, we can now compute the volume bounded by the polygon and rectangular prism.
  // Note that the volume we are interested in is the volume that includes the point v0
  // This can be simply calculated by doing the following:
  //   * For every triangle on the surface of the bounded volume, calculate the volume of the triangular pyramid from
  //     said triangle to vertex v0.
  //   * Sum the absolute volumes of these partials to get the sum
  // This works because the volume formed by intersecting a plane with a cube is guaranteed to be a convex
  // polygon.

  return 0.0f;
}

/*
  left and right should be the coefficients for the hessian normal form of the plane n dot x = d.
  assume the indices are such that 0 -> x, 1 -> y, 2 -> z, 3 -> d
  scale should be the x, y, z sidelength values
  Assumes that the normal of the left and right planes point towards each other. e.g. n_l dot n_r < 0
 */
float volume_in_bounds(
        tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
        Plane p_left,
        Plane p_right,
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

        // Ignore voxels that are outside of the surface we'd like to reconstruct
        if (!inside_surface(tsdf, i, j, k))
          continue;

        IntersectionType left_intersection  = intersect_type(p_left,  i, j, k, scale);
        IntersectionType right_intersection = intersect_type(p_right, i, j, k, scale);

        // Ignore voxels that are outside of the specified planes
        if (left_intersection == IntersectionType::OUTSIDE ||
            right_intersection == IntersectionType::OUTSIDE)
          continue;

        float percentVolume = 0.0f;

        if (left_intersection == IntersectionType::INTERSECTS) {
          // If we intersect with the left plane
          //percentVolume = percent_volume(p_left, i, j, k, scale);
        } else if (right_intersection == IntersectionType::INTERSECTS) {
          // If we intersect with the right plane
          //percentVolume = percent_volume(p_right, i, j, k, scale);
        } else {
          // Only other case is that the voxel lies entirely within the confines of
          // both planes
          percentVolume = 1.0f;
        }
        volume += scale[0] * scale[1] * scale[2] * percentVolume;
    }

    return volume;
}

}
}
