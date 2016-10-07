/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <thread>
#include <tdp/inertial/imu_interface.h>
#include <tdp/inertial/imu_outstream.h>
#include <tdp/inertial/pose_interpolator.h>
#include <tdp/utils/threadedValue.hpp>

namespace tdp {

class ImuInterpolator {
 public:
  ImuInterpolator(tdp::ImuInterface* imu, tdp::ImuOutStream* out = nullptr)  
    : imu_(imu), out_(out), receiveImu_(false), numReceived_(0),
      calibrated_(false),
      gyro_bias_(Eigen::Vector3f::Zero())
  {}            
  ~ImuInterpolator()
  {}

  void Start();
  void Stop();

  void StartRecording() { if (!record_.Get()) record_.Set(true); }
  void StopRecording()  { if (record_.Get()) record_.Set(false); }
  
  tdp::PoseInterpolator Ts_wi_;
 private:
  tdp::ImuInterface* imu_;
  tdp::ImuOutStream* out_;
  tdp::ThreadedValue<bool> record_;
  tdp::ThreadedValue<bool> receiveImu_;
  tdp::ThreadedValue<size_t> numReceived_;
  std::thread receiverThread_;

  bool calibrated_;
  Eigen::Vector3f gyro_bias_;
  Eigen::Vector3f gravity0_;
};

}
