/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tdp/manifold/S.h>
#include <tdp/utils/combinations.h>
#include <tdp/bb/tetrahedron.h>

namespace tdp {

template <typename T>
std::vector<Tetrahedron4D<T>> TessellateS3();
template <typename T>
std::vector<Tetrahedron4D<T>> TessellateS3(const Eigen::Matrix<T,4,1>& north);
template <typename T>
void TessellationTest(std::vector<Tetrahedron4D<T>>& tetrahedra, uint32_t Nsamples);

}
