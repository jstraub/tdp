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
      const Plane plane,
      size_t i, size_t j, size_t k,
      const Vector3fda grid0,
      const Vector3fda dGrid,
      const SE3f T_wG
  );

  // Returns the index in the input array of the vertex that should be labeled v0
  int find_v0(const Plane plane, const Vector3fda* tmp);

  /*
   * Returns the percentage of the volume of the voxel that is on the "positive" side of the
   * plane as defined by the normal of the plane
   */
  float percent_volume(
      const Plane plane,
      size_t i, size_t j, size_t k,
      const Vector3fda grid0,
      const Vector3fda dGrid,
      const SE3f T_wG
  );

  float volume_in_bounds_with_voxel_counting(
      const ManagedHostVolume<TSDFval>& tsdf,
      const Plane p_left,
      const Plane p_right,
      const Vector3fda grid0,
      const Vector3fda dGrid,
      const SE3f T_wG
  );

  float volume_in_bounds_with_tsdf_modification(
      const ManagedHostVolume<TSDFval>& tsdf,
      const Plane p_left,
      const Plane p_right,
      const Vector3fda scale
  );
}
}
