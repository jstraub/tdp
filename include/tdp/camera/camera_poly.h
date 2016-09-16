#pragma once
#include <Eigen/Dense>
#include <tdp/config.h>
#include <tdp/eigen/dense.h>
#include <tdp/camera/camera_base.h>
#include <pangolin/utils/picojson.h>

namespace tdp {

template <class T>
class CameraPoly3 : public CameraBase<T,7,CameraPoly3<T>> {
 public:
  typedef Eigen::Matrix<T,7,1> Parameters;

  const static int NumParams = 7;

  CameraPoly3()
  {}
  // parameters: fu, fv, uc, vc
  CameraPoly3(const Eigen::Matrix<T,4,1>& params) 
  {
    this->params_.Fill(0.);
    this->params_.topRows(4) = params;
  }
  // parameters: fu, fv, uc, vc, p1, p2, p3
  CameraPoly3(const Parameters& params) 
    : CameraBase<T,7,CameraPoly3<T>>(params)
  {}
  CameraPoly3(const CameraPoly3& other) 
    : CameraBase<T,7,CameraPoly3<T>>(other.params_)
  {}
  ~CameraPoly3()
  {}

  TDP_HOST_DEVICE
  Eigen::Matrix<T,2,1> Project(const Eigen::Matrix<T,3,1>& p) const {
    Eigen::Matrix<T,2,1> ray(p(0)/p(2), p(1)/p(2));
    T r = ray.norm();
    T r2 = r*r;
    T r4 = r2*r2;
    T scale = 1 + r2*this->params_(4) + r4*this->params_(5) + r4*r2*this->params_(6);
    return Eigen::Matrix<T,2,1> (
        scale*ray(0)*this->params_(0)+this->params_(2), 
        scale*ray(1)*this->params_(1)+this->params_(3));
  }

  TDP_HOST_DEVICE
  Eigen::Matrix<T,3,1> Unproject(T u, T v, T z) const {
    Eigen::Matrix<T,2,1> ray(
        (u-this->params_(2))/this->params_(0),
        (v-this->params_(3))/this->params_(1));
     
    T k1 = this->params_(4);
    T k2 = this->params_(5);
    T k3 = this->params_(6);
    // Use Newton's method to solve (fixed number of iterations)
    // From Calibu camera_models_poly.h
    T r = ray.norm();
    T ru = r;
    for (int i=0; i < 5; i++) {
      // Common sub-expressions of d, d2
      T ru2 = ru * ru;
      T ru4 = ru2 * ru2;
      T ru6 = ru4 * ru2;
      T pol = k1 * ru2 + k2 * ru4 + k3 * ru6 + 1;
      T pol2 = 2 * ru2 * (k1 + 2 * k2 * ru2 + 3 * k3 * ru4);
      T pol3 = pol + pol2;

      // 1st derivative
      T d = (ru * (pol) - r)  *  2 * pol3;
      // 2nd derivative
      T d2 = (4 * ru * (ru * pol - r) *
              (3 * k1 + 10 * k2 * ru2 + 21 * k3 * ru4) +
              2 * pol3 * pol3);
      // Delta update
      T delta = d / d2;
      ru -= delta;
    }
    //std::cout << ru << " " << r << std::endl;
    T scale = ru / r;
    if (isnan(scale)) scale = 1.;
    //std::cout << ray.transpose() << "; " << scale << " " << z << std::endl;
    return Eigen::Matrix<T,3,1>(ray(0)*scale*z, ray(1)*scale*z, z);
  }

  Eigen::Matrix<T,3,3> GetK() const {
    Eigen::Matrix<T,3,3> K = Eigen::Matrix<T,3,3>::Zero();
    K(0,0) = this->params_(0);
    K(1,1) = this->params_(1);
    K(2,2) = 1.;
    K(0,2) = this->params_(2);
    K(1,2) = this->params_(3);
    return K;
  }

  Eigen::Matrix<T,3,3> GetKinv() const {
    Eigen::Matrix<T,3,3> Kinv = Eigen::Matrix<T,3,3>::Zero();
    Kinv(0,0) = 1./this->params_(0);
    Kinv(1,1) = 1./this->params_(1);
    Kinv(2,2) = 1.;
    Kinv(0,2) = -this->params_(2)/this->params_(0);
    Kinv(1,2) = -this->params_(3)/this->params_(1);
    return Kinv;
  }

  bool FromJson(pangolin::json::value& val){
    if (!val.contains("model"))
      return false;
    if (!val["model"].contains("type"))
      return false;
    if (val["model"]["type"].get<std::string>().compare("poly3") != 0)
      return false;
    this->params_(0) = val["model"]["fu"].get<double>(); 
    this->params_(1) = val["model"]["fv"].get<double>(); 
    this->params_(2) = val["model"]["uc"].get<double>(); 
    this->params_(3) = val["model"]["vc"].get<double>(); 
    this->params_(4) = val["model"]["distortion"][0].get<double>(); 
    this->params_(5) = val["model"]["distortion"][1].get<double>(); 
    this->params_(6) = val["model"]["distortion"][2].get<double>(); 
    return true;
  }

  pangolin::json::value ToJson() const {
    pangolin::json::value val;
    val["model"] = pangolin::json::value();
    val["model"]["fu"] = this->params_(0); 
    val["model"]["fv"] = this->params_(1); 
    val["model"]["uc"] = this->params_(2); 
    val["model"]["vc"] = this->params_(3); 
    val["model"]["distortion"] = pangolin::json::array();
    val["model"]["distortion"].push_back(this->params_(4));
    val["model"]["distortion"].push_back(this->params_(5));
    val["model"]["distortion"].push_back(this->params_(6));
    val["model"]["type"] = "poly3";
    return val;
  }

};


typedef CameraPoly3<float>  CameraPoly3f;
typedef CameraPoly3<double> CameraPoly3d;

template<typename T>
CameraPoly3<T> ScaleCamera(const CameraPoly3<T>& cam, T scale) {
  //Camera<T>::Parameters paramsScaled = cam.params_;
  Eigen::Matrix<T,7,1> paramsScaled = cam.params_;
  paramsScaled(0) *= scale;
  paramsScaled(1) *= scale;
  paramsScaled(2) = (paramsScaled(2)+0.5)*scale-0.5;
  paramsScaled(3) = (paramsScaled(3)+0.5)*scale-0.5;
  paramsScaled(4) *= scale;
  paramsScaled(5) *= scale;
  paramsScaled(6) *= scale;
  return CameraPoly3<T>(paramsScaled);
};
}
