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

std::vector<Tetrahedron4D> TessellateS3();
std::vector<Tetrahedron4D> TessellateS3(const Eigen::Vector4f& north);
void TessellationTest(std::vector<Tetrahedron4D>& tetrahedra, uint32_t Nsamples);

}
