#include <vector>
#include <algorithm>
#include <tdp/testing/testing.h>
#include <tdp/directional/hist.h>
#include <tdp/directional/geodesic_grid.h>

using namespace tdp;

TEST(areas, GeoGrid1) {

  GeodesicGrid<1> grid;
  ASSERT_EQ(grid.NTri(), 20);
  auto end = std::unique(grid.tri_areas_.begin(), grid.tri_areas_.end(), 
      [](float l, float r){ return fabs(l-r)<1e-6; });
  std::cout << "areas ";
  for (auto areaIt = grid.tri_areas_.begin(); areaIt != end; areaIt++ )
    std::cout << *areaIt << " ";
  std::cout << std::endl;

  for (size_t i=0; i<grid.NTri(); ++i) {
    ASSERT_LT(grid.tri_[i](0), grid.pts_.size());
    ASSERT_LT(grid.tri_[i](1), grid.pts_.size());
    ASSERT_LT(grid.tri_[i](2), grid.pts_.size());

    Eigen::Vector3f n = (grid.pts_[grid.tri_[i](1)] -grid.pts_[grid.tri_[i](0)]
        ).cross(grid.pts_[grid.tri_[i](2)]- grid.pts_[grid.tri_[i](0)]);
    n /= n.norm();
//    std::cout << n.dot(grid.pts_[grid.tri_[i](0)]) << std::endl;
    ASSERT_GE(n.dot(grid.pts_[grid.tri_[i](0)]), 0.);
  }
}

TEST(areas, GeoGrid2) {

  GeodesicGrid<2> grid;
  ASSERT_EQ(grid.NTri(), 80);
  std::sort(grid.tri_areas_.begin(), grid.tri_areas_.end());
  auto end = std::unique(grid.tri_areas_.begin(), grid.tri_areas_.end(), 
      [](float l, float r) { return fabs(l-r)<1e-6; });
  std::cout << "areas ";
  for (auto areaIt = grid.tri_areas_.begin(); areaIt != end; areaIt++ )
    std::cout << *areaIt << " ";
  std::cout << std::endl;

  for (size_t i=1; i<grid.tri_lvls_.size(); ++i) {
    for (size_t j=grid.tri_lvls_[i-1]; j<grid.tri_lvls_[i]; ++j) {
      ASSERT_LT(grid.tri_[j](0), grid.pts_.size());
      ASSERT_LT(grid.tri_[j](1), grid.pts_.size());
      ASSERT_LT(grid.tri_[j](2), grid.pts_.size());

      Eigen::Vector3f n = (grid.pts_[grid.tri_[j](1)] -grid.pts_[grid.tri_[j](0)]
          ).cross(grid.pts_[grid.tri_[j](2)]- grid.pts_[grid.tri_[j](0)]);
      n /= n.norm();
      ASSERT_GE(n.dot(grid.pts_[grid.tri_[j](0)]), 0.);
    }
  }
}

TEST(areas, GeoGrid3) {

  GeodesicGrid<3> grid;
  ASSERT_EQ(grid.NTri(), 320);
  std::sort(grid.tri_areas_.begin(), grid.tri_areas_.end());
  auto end = std::unique(grid.tri_areas_.begin(), grid.tri_areas_.end(), 
      [](float l, float r) { return fabs(l-r)<1e-6; });
  std::cout << "areas ";
  for (auto areaIt = grid.tri_areas_.begin(); areaIt != end; areaIt++ )
    std::cout << *areaIt << " ";
  std::cout << std::endl;

  for (size_t i=1; i<grid.tri_lvls_.size(); ++i) {
    for (size_t j=grid.tri_lvls_[i-1]; j<grid.tri_lvls_[i]; ++j) {
      ASSERT_LT(grid.tri_[j](0), grid.pts_.size());
      ASSERT_LT(grid.tri_[j](1), grid.pts_.size());
      ASSERT_LT(grid.tri_[j](2), grid.pts_.size());

      Eigen::Vector3f n = (grid.pts_[grid.tri_[j](1)] -grid.pts_[grid.tri_[j](0)]
          ).cross(grid.pts_[grid.tri_[j](2)]- grid.pts_[grid.tri_[j](0)]);
      n /= n.norm();
      ASSERT_GE(n.dot(grid.pts_[grid.tri_[j](0)]), 0.);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

