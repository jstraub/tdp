#include <tdp/manifold/SE3.h>

namespace tdp {

template class SE3<float ,SO3<float ,Eigen::DontAlign>, Eigen::DontAlign>;
template class SE3<double,SO3<double,Eigen::DontAlign>, Eigen::DontAlign>;

template class SE3<float ,SO3<float >,Eigen::DontAlign>;
template class SE3<double,SO3<double>,Eigen::DontAlign>;

}
