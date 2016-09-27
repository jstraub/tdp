#pragma once

#include <tdp/inertial/imu_obs.h>
#include <pangolin/utils/picojson.h>

namespace tdp {

class ImuInterface {
 public:
  ImuInterface() {}
  virtual ~ImuInterface() {}

  virtual bool GrabNext(ImuObs& obs) = 0;
  virtual bool GrabNewest(ImuObs& obs) = 0;

  virtual void Start() = 0;
  virtual void Stop() = 0;

  virtual pangolin::json::value GetProperties() = 0;

 private:

};

}
