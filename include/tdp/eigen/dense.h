/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <stdint.h>
#include <Eigen/Dense>

namespace tdp {

typedef Eigen::Matrix<uint8_t,3,1> Vector3b;
typedef Eigen::Matrix<uint8_t,3,1,Eigen::DontAlign> Vector3bda;
typedef Eigen::Matrix<uint8_t,4,1,Eigen::DontAlign> Vector4bda;


typedef Eigen::Matrix<float,29,1,Eigen::DontAlign> Vector29fda;
typedef Eigen::Matrix<float,11,1,Eigen::DontAlign> Vector11fda;
typedef Eigen::Matrix<float,10,1,Eigen::DontAlign> Vector10fda;
typedef Eigen::Matrix<float,7,1,Eigen::DontAlign> Vector7fda;
typedef Eigen::Matrix<float,6,1,Eigen::DontAlign> Vector6fda;
typedef Eigen::Matrix<float,5,1,Eigen::DontAlign> Vector5fda;
typedef Eigen::Matrix<float,4,1,Eigen::DontAlign> Vector4fda;
typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> Vector3fda;
typedef Eigen::Matrix<float,2,1,Eigen::DontAlign> Vector2fda;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::DontAlign> VectorXfda;

typedef Eigen::Matrix<float,6,6,Eigen::DontAlign> Matrix6fda;
typedef Eigen::Matrix<float,3,3,Eigen::DontAlign> Matrix3fda;
typedef Eigen::Matrix<float,2,2,Eigen::DontAlign> Matrix2fda;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic/*, Eigen::DontAlign*/> MatrixXfda;

typedef Eigen::Matrix<int32_t,2,1,Eigen::DontAlign> Vector2ida;
typedef Eigen::Matrix<int32_t,3,1,Eigen::DontAlign> Vector3ida;
typedef Eigen::Matrix<int32_t,4,1,Eigen::DontAlign> Vector4ida;
typedef Eigen::Matrix<int32_t,5,1,Eigen::DontAlign> Vector5ida;

typedef Eigen::Matrix<uint32_t,3,1,Eigen::DontAlign> Vector3uda;

// for BRIEF features
typedef Eigen::Matrix<uint32_t,4,1,Eigen::DontAlign> Vector4uda;
typedef Eigen::Matrix<uint32_t,8,1,Eigen::DontAlign> Vector8uda;
typedef Eigen::Matrix<uint32_t,16,1,Eigen::DontAlign> Vector16uda;

}
