#include <iostream>
#include <cmath>

#include <tdp/reconstruction/volumeReconstruction.h>
#include "test.h"

void test_find_v0_helper(int testNum, tdp::Reconstruction::Plane plane, Eigen::Vector3f *vectors, size_t expected) {
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
    Eigen::Vector3f scale(1, 1, 1);

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

    test_find_v0_helper(1, tdp::Reconstruction::Plane( 1,  1,  1,  sqrt(3.0)/3), tmp, 6);
    test_find_v0_helper(2, tdp::Reconstruction::Plane(-1, -1, -1, -sqrt(3.0)/3), tmp, 0);
}

void runtests() {
    test_find_v0();
}