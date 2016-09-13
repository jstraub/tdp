
#pragma once
#include <tdp/inertial/imu_interface.h>

namespace tdp {

class Imu3DMGX3_45 : public ImuInterface Best{
 public:

  Imu3DMGX3_45() {}
  virtual ~Imu3DMGX3_45() {}

  virtual void GrabNext(ImuObs& obs);

  virtual void Start();
  virtual void Stop();

 private:

};

}
