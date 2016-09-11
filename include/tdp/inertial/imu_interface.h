#pragma once

#include <tdp/inertial/imu_obs.h>

namespace tdp {

class ImuInterface {
 public:
  ImuInterface() {}
  virtual ~ImuInterface() {}

  virtual void GrabNext(ImuObs& obs) = 0;

  virtual void Start() = 0;
  virtual void Stop() = 0;

 private:

};

}
