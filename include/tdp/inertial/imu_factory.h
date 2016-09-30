#pragma once

#include <string>
#include <iostream>
#include <pangolin/utils/uri.h>
#include <pangolin/utils/file_utils.h>
#include <tdp/inertial/imu_interface.h>

#ifdef ASIO_FOUND
#  include <tdp/drivers/inertial/3dmgx3_45.h>
#endif

#include <tdp/drivers/inertial/imu_pango.h>

namespace tdp {

ImuInterface* OpenImu(const std::string& uri_str) {
  ImuInterface* imu = nullptr;
  const pangolin::Uri uri = pangolin::ParseUri(uri_str);
  
  if (!uri.scheme.compare("file")) {
    const bool realtime = uri.Contains("realtime");
    const std::string path = pangolin::PathExpand(uri.url);
    imu = new ImuPango(path, realtime);
  }
#ifdef ASIO_FOUND
  else if (!uri.scheme.compare("3dmgx3")) {
    const std::string port = uri.Get<std::string>("port","/dev/ttyACM0");
    const int rate = uri.Get<int>("rate",10);
    imu = new Imu3DMGX3_45(port, rate);
  } 
#endif
  else {
    std::cerr << "IMU uri not recognized: " << uri.scheme << std::endl;
  }
  return imu;
}

}
