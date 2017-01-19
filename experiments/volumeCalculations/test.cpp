#include <iostream>
#include <cmath>

#include <tdp/eigen/dense.h>
#include <tdp/reconstruction/volumeReconstruction.h>
#include "test.h"

void test_find_v0_helper(int testNum, tdp::Reconstruction::Plane plane, tdp::Vector3fda *vectors, size_t expected) {
    std::cout << "\t" << testNum << ": ";

    size_t actual = tdp::Reconstruction::find_v0(plane, vectors);

    if (actual != expected) {
        std::cout << "FAIL" << std::endl;
        std::cout << "\t\tExpected: " << expected << std::endl;
        std::cout << "\t\tActual:   " << actual << std::endl;
    } else {
        std::cout << "PASS" << std::endl;
    }
}

void test_find_v0() {
    std::cout << "Test: Find V0" << std::endl;
    bool fail = false;

    float i = 0, j = 0, k = 0;
    tdp::Vector3fda scale(1, 1, 1);

    tdp::Vector3fda tmp[8] = {
        tdp::Vector3fda((i    ) * scale(0), (j    ) * scale(1), (k    ) * scale(2)),
        tdp::Vector3fda((i + 1) * scale(0), (j    ) * scale(1), (k    ) * scale(2)),
        tdp::Vector3fda((i + 1) * scale(0), (j + 1) * scale(1), (k    ) * scale(2)),
        tdp::Vector3fda((i    ) * scale(0), (j + 1) * scale(1), (k    ) * scale(2)),
        tdp::Vector3fda((i    ) * scale(0), (j    ) * scale(1), (k + 1) * scale(2)),
        tdp::Vector3fda((i + 1) * scale(0), (j    ) * scale(1), (k + 1) * scale(2)),
        tdp::Vector3fda((i + 1) * scale(0), (j + 1) * scale(1), (k + 1) * scale(2)),
        tdp::Vector3fda((i    ) * scale(0), (j + 1) * scale(1), (k + 1) * scale(2))
    };

    test_find_v0_helper(1, tdp::Reconstruction::Plane( 1,  1,  1,  sqrt(3.0)/3), tmp, 6);
    test_find_v0_helper(2, tdp::Reconstruction::Plane(-1, -1, -1, -sqrt(3.0)/3), tmp, 0);
}

void test_check_plane_dist_to_origin(int testNum, float x, float y, float z, float d, float expected) {
    std::cout << "\t" << testNum << ": ";
    tdp::Reconstruction::Plane plane(x, y, z, d);

    float actual = plane.distance_to(tdp::Vector3fda(0,0,0));

    if (abs(expected - actual) < 1e-6) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      std::cout << "\t\tExpected: " << expected << std::endl;
      std::cout << "\t\tActual:   " << actual << std::endl;
    }
}

void test_check_plane_distances() {
    std::cout << "Test: Plane distance" << std::endl;
    test_check_plane_dist_to_origin(1, 1, 1, 1, 1, -1/sqrt(3));
    test_check_plane_dist_to_origin(2, -1, -1, -1, -1, 1/sqrt(3));
    test_check_plane_dist_to_origin(3, 1, 1, 1, -1, 1/sqrt(3));
}

void runtests() {
    test_find_v0();
    test_check_plane_distances();
}
