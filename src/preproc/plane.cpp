
#include <tdp/preproc/plane.h>

namespace tdp {

void Plane::AddObs(const Vector3fda& p, const Vector3fda& n) {
  float wNew = (w_*n_+n).norm(); 
  float dw = wNew-w_; 
  n_ = (n_*w_ + n).normalized();
  p_ += n.dot(dw*p-w_*p_)/wNew * n;
  w_ = std::min(100.f, wNew);
}

void Plane::AddObs(const Vector3fda& p, const Vector3fda& n, 
    const Vector3bda& rgb) {
  float wNew = (w_*n_+n).norm(); 
  float dw = wNew-w_; 
  p_ = (p_*w_ + dw*p)/wNew;
  n_ = (n_*w_ + dw*n).normalized();
  rgb_ = ((rgb_.cast<float>()*w_ + dw*rgb.cast<float>())/wNew).cast<uint8_t>();
  w_ = std::min(100.f, wNew);
}

tdp::SE3f Plane::LocalCosy() {
  Eigen::Matrix3f R = tdp::OrthonormalizeFromYZ(
      dir_, n_);
  return tdp::SE3f(R, p_); 
}


bool Plane::Close(const Plane& other, float dotThr, float distThr, 
    float p2plThr) {
  if ((p_ - other.p_).norm() < distThr) {
    if (fabs(n_.dot(other.n_)) > dotThr) {
      if (fabs(n_.dot(p_ - other.p_)) < p2plThr) {
        return true;
      }
    }
  }
  return false;
}

}
