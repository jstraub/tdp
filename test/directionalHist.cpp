#include <tdp/testing/testing.h>
#include <tdp/directional/hist.h>
#include <tdp/directional/geodesic_grid.h>

using namespace tdp;

TEST(areas, GeoGrid) {

  GeodesicGrid<1> grid1;
  for (size_t i=0; i<grid1.NTri(); ++i) {
    std::cout << gird1.tri_areas_[i] << std::endl;
  }

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

