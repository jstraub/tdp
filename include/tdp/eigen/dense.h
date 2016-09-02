#pragma once
#include <Eigen/Dense>

namespace tdp {
typedef Eigen::Matrix<float,4,1,Eigen::DontAlign> Vector4fda;
typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> Vector3fda;
typedef Eigen::Matrix<float,2,1,Eigen::DontAlign> Vector2fda;

typedef Eigen::Matrix<float,3,3,Eigen::DontAlign> Matrix3fda;
typedef Eigen::Matrix<float,2,2,Eigen::DontAlign> Matrix2fda;
}
