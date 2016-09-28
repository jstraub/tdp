/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#pragma once
#include <thread>
#include <mutex>

#include <Eigen/Dense>
#include <tdp/manifold/SE3.h>

namespace tdp {

/// Thread-save pose interpolator
/// All times t are in nano seconds
class PoseInterpolator {
 public:  
  PoseInterpolator()
  {}
  ~PoseInterpolator()
  {}

  /// Add a new <t,Pose> observation; 
  /// IMPORTANT: the assumption is that poses come in in chronological
  /// order.
  void Add(int64_t t, const SE3f& T) {
    std::lock_guard<std::mutex> lock(mutex_);
    ts_.push_back(t);
    Ts_.push_back(T);
  }

  void Add(int64_t t, const Eigen::Matrix<float,6,1>& se3, int64_t dt_ns = -1) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t end = ts_.size()-1;
    float dt = (t-ts_[end])*1e-9;
    if (dt_ns >= 0) dt = dt_ns*1e-9;
    Ts_.push_back(Ts_[end].Exp(se3*dt));
    ts_.push_back(t);
  }

  SE3f operator[](int64_t t) {
    std::lock_guard<std::mutex> lock(mutex_);
//    std::cout <<  "PoseInterpolator: getting pose at " << t 
//      << ": " << ts_.size() << ", " << Ts_.size() 
//      << "; " << ts_[0]  << ", " << ts_[ts_.size()-1]
//      << std::endl;
    size_t i = 0;
    // lock while we are looking for index then release
    for(i=0; i<ts_.size(); ++i) if (ts_[i]-t >=0) break;
    if (0<i && i<ts_.size()-1) {
      float factor = (float)(ts_[i]-t)/(float)(ts_[i]-ts_[i-1]);
//      std::cout <<  "PoseInterpolator: interpolating " << i << ", " << (ts_[i]-t)*1e-9 << " " << factor << std::endl;
      return Ts_[i-1].Exp(Ts_[i-1].Log(Ts_[i]) * factor);
    } else if (i==ts_.size()) {
      float factor = (float)(t-ts_[i-1])/(float)(ts_[i-1]-ts_[i-2]);
//      std::cout <<  "PoseInterpolator: extrapolating " << i << ", " << (t-ts_[i-1])*1e-9  << " " <<  factor << std::endl;
      return Ts_[i-1].Exp(Ts_[i-2].Log(Ts_[i-1])*factor);
    } else if (Ts_.size()>0) {
      return Ts_[i];
    } else {
      return SE3f();
    }
  }

 private:
  std::vector<int64_t> ts_; // time stamp in nano seconds
  std::vector<SE3f> Ts_;
  std::mutex mutex_;
};


}
