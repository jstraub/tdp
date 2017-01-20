#include <iostream>
#include <algorithm>

#include <tdp/reconstruction/volumeReconstruction.h>

namespace tdp {
namespace Reconstruction {

static Vector3fda tsdf_point_to_real_space_point(
              size_t i,
              size_t j,
              size_t k,
              Vector3fda grid0,
              Vector3fda dGrid,
              SE3f T_wG) {
  Vector3fda base(i * dGrid(0), j * dGrid(1), k * dGrid(2));
  return T_wG * (base + grid0);
}

// Returns true if a voxel is completely inside the surface
bool inside_surface(const ManagedHostVolume<TSDFval>& tsdf, size_t x, size_t y, size_t z) {
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

IntersectionType intersect_type(const Plane plane,
                                const Vector3fda corner1,
                                const Vector3fda corner2) {
  float x[2] = {corner1(0), corner2(0)};
  float y[2] = {corner1(1), corner2(1)};
  float z[2] = {corner1(2), corner2(2)};

  bool hasInside = false, hasOutside = false;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        Vector3fda p(x[i], y[j], z[k]);

        // Calculate the distance to the plane from each corner
        float out = plane.distance_to(p);

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

// TODO: Slow by a factor of 8 (each point calculated 8 times)
IntersectionType intersect_type(const Plane plane,
                                size_t i,
                                size_t j,
                                size_t k,
                                const Vector3fda grid0,
                                const Vector3fda dGrid,
                                const SE3f T_wG) {
  bool hasInside = false, hasOutside = false;

  for (int dx = 0; dx <= 1; dx++)
    for (int dy = 0; dy <= 1; dy++)
      for(int dz = 0; dz <= 1; dz++) {
        Vector3fda x = tsdf_point_to_real_space_point(i + dx, j + dy, k + dz, grid0, dGrid, T_wG);

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

int find_v0(const Plane plane, const Vector3fda* tmp) {
  // Finds the point furthest from the plane given it's normal
  int index = 0;
  float minVal = plane.distance_to(tmp[0]);

  for (int i = 1; i < 8; i++) {
    float val = plane.distance_to(tmp[i]);

    if (val < minVal) {
      minVal = val;
      index = i;
    }
  }

  return index;
}

static bool good_intersection(const Plane plane,
                             const Vector3fda v,
                             const Vector3fda e) {

  float denominator = plane.unit_normal().dot(e);
  if (denominator == 0) {
    return false;
  }
  float numerator = -plane.distance_to(v);
  float lambda = numerator / denominator;
  return lambda >= 0 && lambda <= 1;
}
static inline Vector3fda get_intersection(const Plane plane,
                                   const Vector3fda v,
                                   const Vector3fda e) {
  return v + (-plane.distance_to(v) / plane.unit_normal().dot(e)) * e;
}
static Vector3fda find_intersection_along_edges_with_right_hand_rule(
        const Plane plane,
        const Vector3fda v0,
        const Vector3fda e0,
        const Vector3fda v1,
        const Vector3fda e1,
        const Vector3fda v2,
        const Vector3fda e2) {
  float lambda;

  if (good_intersection(plane, v0, e0)) {
    return get_intersection(plane, v0, e0);
  } else if (good_intersection(plane, v1, e1)) {
    return get_intersection(plane, v1, e1);
  } else if (good_intersection(plane, v2, e2)) {
    return get_intersection(plane, v2, e2);
  } else {
    std::cerr << "ERROR: No Intersection found" << std::endl;
    exit(1);
  }
}

// Corners must be opposite and polygon must be of length 6
// the polygon found will have the same orientation as the original plane
void get_vertices_of_intersection(Vector3fda* polygon,
                                  const Plane plane,
                                  const Vector3fda corner1,
                                  const Vector3fda corner2) {
  // For explanatory reasons, let us number the vertices of a unit cube from 0 - 7 as follows
  // (x    , y    , z    ) -> 0
  // (x + 1, y    , z    ) -> 1
  // (x + 1, y + 1, z    ) -> 2
  // (x    , y + 1, z    ) -> 3
  // (x    ,      , z + 1) -> 4
  // (x + 1, y    , z + 1) -> 5
  // (x + 1, y + 1, z + 1) -> 6
  // (x    , y + 1, z + 1) -> 7
  float minX = std::min(corner1(0), corner2(0)),
        minY = std::min(corner1(1), corner2(1)),
        minZ = std::min(corner1(2), corner2(2));
  float maxX = std::max(corner1(0), corner2(0)),
        maxY = std::max(corner1(1), corner2(1)),
        maxZ = std::max(corner1(2), corner2(2));

  Vector3fda tmp[8] = {
    Vector3fda(minX, minY, minZ),
    Vector3fda(maxX, minY, minZ),
    Vector3fda(maxX, maxY, minZ),
    Vector3fda(minX, maxY, minZ),
    Vector3fda(minX, minY, maxZ),
    Vector3fda(maxX, minY, maxZ),
    Vector3fda(maxX, maxY, maxZ),
    Vector3fda(minX, maxY, maxZ)
  };

  // then we just need to figure out which vertex is "first" and then from that we have a deterministic
  // mapping from numbers (0-7) -> (v0 - v7).

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

  int index = find_v0(plane, tmp);
  Vector3fda v[8];
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

  Vector3fda E01 = v[1] - v[0];
  Vector3fda E14 = v[4] - v[1];
  Vector3fda E47 = v[7] - v[4];
  Vector3fda E15 = v[5] - v[1];
  Vector3fda E02 = v[2] - v[0];
  Vector3fda E25 = v[5] - v[2];
  Vector3fda E57 = v[7] - v[5];
  Vector3fda E26 = v[6] - v[2];
  Vector3fda E03 = v[3] - v[0];
  Vector3fda E36 = v[6] - v[3];
  Vector3fda E67 = v[7] - v[6];
  Vector3fda E34 = v[4] - v[3];

  polygon[0] = find_intersection_along_edges_with_right_hand_rule(plane, v[0], E01, v[1], E14, v[4], E47);
  polygon[2] = find_intersection_along_edges_with_right_hand_rule(plane, v[0], E02, v[2], E25, v[5], E57);
  polygon[4] = find_intersection_along_edges_with_right_hand_rule(plane, v[0], E03, v[3], E36, v[6], E67);

  polygon[1] = good_intersection(plane, v[1], E15) ? get_intersection(plane, v[1], E15) : polygon[0];
  polygon[3] = good_intersection(plane, v[2], E26) ? get_intersection(plane, v[2], E26) : polygon[2];
  polygon[5] = good_intersection(plane, v[3], E34) ? get_intersection(plane, v[3], E34) : polygon[4];
}


/*
 * Returns the percentage of the volume of the voxel that is on the "positive" side of the
 * plane as defined by the normal of the plane
 */
float percent_volume(const Plane plane,
                    size_t i,
                    size_t j,
                    size_t k,
                    const Vector3fda grid0,
                    const Vector3fda dGrid,
                    const SE3f T_wG) {
  Vector3fda corner1 = tsdf_point_to_real_space_point(i    , j    , k    , grid0, dGrid, T_wG);
  Vector3fda corner2 = tsdf_point_to_real_space_point(i + 1, j + 1, k + 1, grid0, dGrid, T_wG);

  Vector3fda p[6];
  get_vertices_of_intersection(p, plane, corner1, corner2);

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
float volume_in_bounds_with_voxel_counting(
        const ManagedHostVolume<TSDFval>& tsdf,
        const Plane p_left,
        const Plane p_right,
        const Vector3fda grid0,
        const Vector3fda dGrid,
        const SE3f T_wG
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

        IntersectionType left_intersection  = intersect_type(p_left,  i, j, k, grid0, dGrid, T_wG);
        IntersectionType right_intersection = intersect_type(p_right, i, j, k, grid0, dGrid, T_wG);

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
        volume += dGrid(0) * dGrid(1) * dGrid(2) * percentVolume;
    }

    return volume;
}

float volume_in_bounds_with_tsdf_modification(
      const ManagedHostVolume<TSDFval>& tsdf,
      const Plane p_left,
      const Plane p_right,
      const Vector3fda scale) {
  ManagedHostVolume<TSDFval> copy(tsdf.w_, tsdf.h_, tsdf.d_);
  for (size_t k = 0; k < tsdf.d_; k++)
    for (size_t j = 0; j < tsdf.h_; j++)
      for (size_t i = 0; i < tsdf.w_; i++) {
        // Make a copy of each element
        copy(i, j, k) = tsdf(i, j, k);

        Vector3fda point(0, 0, 0);

        copy(i, j, k).f = std::max(
          {copy(i, j, k).f, p_left.distance_to(point), p_right.distance_to(point)},
          [](const float& i1, const float& i2) {
            return i1 < i2;
          });

      }

  return 0;
}

}
}
