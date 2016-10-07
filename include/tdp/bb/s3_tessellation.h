/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "manifold/S.h"
#include "bbTrans/combinations.h"
#include "bbTrans/tetrahedron.h"

namespace bb {

std::vector<Tetrahedron4D> TessellateS3();
std::vector<Tetrahedron4D> TessellateS3(const Eigen::Vector4d& north);
void TessellationTest(std::vector<Tetrahedron4D>& tetrahedra, uint32_t Nsamples);

}
