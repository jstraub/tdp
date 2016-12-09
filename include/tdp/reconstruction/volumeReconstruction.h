#pragma once
#include <tdp/eigen/dense.h>
#include <tdp/tsdf/tsdf.h>

#include <tdp/reconstruction/plane.h>

namespace tdp {
namespace Reconstruction {

  // Enumeration used for defining how a voxel is situated with respect to
  // a voxel
  enum class IntersectionType {
      OUTSIDE,
      INSIDE,
      INTERSECTS
  };

  // Returns the type of intersection of the voxel with the plane given.
  //  * The plane is defined by the inputs 'normal' and 'd'. Normal is the unit normal vector
  //    of the plane while 'd' is the signed distance (wrt plane normal) of the plane from the origin
  //  * The voxel is defined by the triple (i, j, k) such that the other corners are part
  //    of the set {i, i + 1} X {j, j + 1} X {k, k + 1}.
  //  * The input scale lets us transform the voxel into the same coordinate system as the
  //    input plane
  IntersectionType intersect_type(
      Plane plane,
      size_t i, size_t j, size_t k,
      Eigen::Vector3f scale
  );

  // Returns the index in the input array of the vertex that should be labeled v0
  int find_v0(Plane plane, Eigen::Vector3f* tmp);

  /*
   * Returns the percentage of the volume of the voxel that is on the "positive" side of the
   * plane as defined by the normal of the plane
   */
  float percent_volume(
      Plane plane,
      size_t i, size_t j, size_t k,
      Eigen::Vector3f scale
  );

  float volume_in_bounds(
      tdp::ManagedHostVolume<tdp::TSDFval>& tsdf,
      Plane p_left,
      Plane p_right,
      Eigen::Vector3f scale
  );

}
}
