#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include <pangolin/utils/picojson.h>
#include <pangolin/image/image_io.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/video/video_record_repeat.h>

#include <tdp/camera/camera.h>
#include <tdp/cuda/cuda.h>
#include <tdp/data/allocator.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/std_vector.h>
#include <tdp/gui/gui.hpp>
#include <tdp/manifold/SE3.h>

namespace tdp {

void CollectRGB(const std::vector<int32_t>& rgbdStream2cam, const GUI&
    gui, size_t wSingle, size_t hSingle,
    Image<Vector3bda>& rgb, cudaMemcpyType type) {

  for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
    Image<Vector3bda> rgbStream;
    if (!gui.ImageRGB(rgbStream, sId)) continue;
    int32_t cId = rgbdStream2cam[sId]; 
    Image<Vector3bda> rgb_i = rgb.GetRoi(0,cId*hSingle,
        wSingle, hSingle);
    rgb_i.CopyFrom(rgbStream,type);
  }
}

void CollectD(const std::vector<int32_t>& rgbdStream2cam, 
    const Rig& rig,
    const GUI& gui, size_t wSingle, size_t hSingle,
    float dMin, float dMax,
    Image<float>& cuDraw,
    Image<float>& cuD) {

  for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
    tdp::Image<uint16_t> dStream;
    if (!gui.ImageD(dStream, sId)) continue;
    int32_t cId = rgbdStream2cam[sId]; 
    tdp::Image<uint16_t> cuDraw_i = cuDraw.GetRoi(0,cId*hSingle,
        wSingle, hSingle);
    cuDraw_i.CopyFrom(dStream,cudaMemcpyHostToDevice);
    // convert depth image from uint16_t to float [m]
    tdp::Image<float> cuD_i = cuD.GetRoi(0, cId*hSingle, 
        wSingle, hSingle);
    //float depthSensorScale = depthSensor1Scale;
    //if (cId==1) depthSensorScale = depthSensor2Scale;
    //if (cId==2) depthSensorScale = depthSensor3Scale;
    if (rig.depthScales_.size() > cId) {
      float a = rig.scaleVsDepths_[cId](0);
      float b = rig.scaleVsDepths_[cId](1);
      // TODO: dont need to load this every time
      cuScale.CopyFrom(rig.depthScales_[cId],cudaMemcpyHostToDevice);
      tdp::ConvertDepthGpu(cuDraw_i, cuD_i, cuScale, a, b, dMin, dMax);
      //} else {
      //  tdp::ConvertDepthGpu(cuDraw_i, cuD_i, depthSensorScale, dMin, dMax);
  }
  }
}


template <class Cam>
struct Rig {

  ~Rig() {
    for (size_t i=0; i<depthScales_.size(); ++i) {
      delete[] depthScales_[i].ptr_;
    }
  }

  bool FromFile(std::string pathToConfig, bool verbose) {
    pangolin::json::value file_json(pangolin::json::object_type,true); 
    std::ifstream f(pathToConfig);
    if (f.is_open()) {
      std::string err = pangolin::json::parse(file_json,f);
      if (!err.empty()) {
        std::cout << file_json.serialize(true) << std::endl;
        if (file_json.size() > 0) {
          std::cout << "found " << file_json.size() << " elements" << std::endl ;
          depthScales_.reserve(file_json.size());
          for (size_t i=0; i<file_json.size(); ++i) {
            if (file_json[i].contains("camera")) {
              // serial number
              if (file_json[i]["camera"].contains("serialNumber")) {
                serials_.push_back(
                    file_json[i]["camera"]["serialNumber"].get<std::string>());
                if (verbose) 
                  std::cout << "Serial ID: " << serials_.back() 
                    << std::endl;
              }
              Cam cam;
              if (cam.FromJson(file_json[i]["camera"])) {
                if (verbose) 
                  std::cout << "found camera model" << std::endl ;
              }
              cams_.push_back(cam);
              if (file_json[i]["camera"].contains("depthScale")) {
                std::string path = file_json[i]["camera"]["depthScale"].get<std::string>();
                depthScalePaths_.push_back(path);
                if (pangolin::FileExists(path)) {
                  pangolin::TypedImage scale8bit = pangolin::LoadImage(path);
                  size_t w = scale8bit.w/4;
                  size_t h = scale8bit.h;
                  std::cout << "w x h: " << w << "x" << h << std::endl;
                  Image<float> scaleWrap(w,h,w*sizeof(float),
                      (float*)scale8bit.ptr);
                  depthScales_.push_back(tdp::Image<float>(w,h,w*sizeof(float),
                        CpuAllocator<float>::construct(w*h)));
                  depthScales_[depthScales_.size()-1].CopyFrom(scaleWrap,
                      cudaMemcpyHostToHost);
                  std::cout << "found and loaded depth scale file"
                    << depthScales_[depthScales_.size()-1].ptr_ << std::endl;
//                  ManagedDeviceImage<float> cuScale(w,h);
//                  cuScale.CopyFrom(depthScales_[depthScales_.size()-1],
//                      cudaMemcpyHostToDevice);
                }
              }
              if (file_json[i]["camera"].contains("depthScaleVsDepthModel")) {
                scaleVsDepths_.push_back(Eigen::Vector2f(
                  file_json[i]["camera"]["depthScaleVsDepthModel"][0].get<double>(),
                  file_json[i]["camera"]["depthScaleVsDepthModel"][1].get<double>()));
              }
              SE3f T_rc;
              if (file_json[i]["camera"].contains("T_rc")) {
                pangolin::json::value t_json =
                  file_json[i]["camera"]["T_rc"]["t_xyz"];
                Eigen::Vector3f t(
                    t_json[0].get<double>(),
                    t_json[1].get<double>(),
                    t_json[2].get<double>());

                Eigen::Matrix3f R_rc;
                if (file_json[i]["camera"]["T_rc"].contains("q_wxyz")) {
                  pangolin::json::value q_json =
                    file_json[i]["camera"]["T_rc"]["q_wxyz"];
                  Eigen::Quaternionf q(q_json[0].get<double>(),
                      q_json[1].get<double>(),
                      q_json[2].get<double>(),
                      q_json[3].get<double>());
                  R_rc = q.toRotationMatrix();
                } else if (file_json[i]["camera"]["T_rc"].contains("R_3x3")) {
                  pangolin::json::value R_json =
                    file_json[i]["camera"]["T_rc"]["R_3x3"];
                  R_rc << R_json[0].get<double>(), R_json[1].get<double>(), 
                       R_json[2].get<double>(), 
                  R_json[3].get<double>(), R_json[4].get<double>(), 
                       R_json[5].get<double>(), 
                  R_json[6].get<double>(), R_json[7].get<double>(), 
                       R_json[8].get<double>();
                }
                T_rc = SE3f(R_rc, t);
                if (verbose) 
                  std::cout << "found T_rc" << std::endl << T_rc << std::endl;
              }
              T_rcs_.push_back(T_rc);
            }
          }
        } else {
          std::cerr << "error json file seems empty"  << std::endl
            << file_json.serialize(true) << std::endl;
          return false;
        }
      } else {
        std::cerr << "error reading json file: " << err << std::endl;
        return false;
      }
    } else {
      std::cerr << "couldnt open file: " << pathToConfig << std::endl;
      return false;
    }
    config_ = file_json;
    return true;
  }

  bool ToFile(std::string pathToConfig, bool verbose) {
    return false;
  }

  // camera to rig transformations
  std::vector<SE3f> T_rcs_; 
  // cameras
  std::vector<Cam> cams_;
  // depth scale calibration images
  std::vector<std::string> depthScalePaths_;
  std::vector<Image<float>> depthScales_;
  // depth scale scaling model as a function of depth
  eigen_vector<Eigen::Vector2f> scaleVsDepths_;

  // camera serial IDs
  std::vector<std::string> serials_;
  // raw properties
  pangolin::json::value config_;
};

/// Uses serial number in openni device props and rig to find
/// correspondences.
template<class Cam>
bool CorrespondOpenniStreams2Cams(
    const std::vector<pangolin::VideoInterface*>& streams,
    const tdp::Rig<Cam>& rig,
    std::vector<int32_t>& rgbStream2cam,
    std::vector<int32_t>& dStream2cam,
    std::vector<int32_t>& rgbdStream2cam
    ) {

  rgbStream2cam.clear();
  dStream2cam.clear();
  rgbdStream2cam.clear();
  
  pangolin::json::value devProps = pangolin::GetVideoDeviceProperties(streams[0]);

  if (! devProps.contains("openni") ) 
    return false;
  pangolin::json::value jsDevices = devProps["openni"]["devices"];

  for (size_t i=0; i<jsDevices.size(); ++i) {
    std::string serial = jsDevices[i]["ONI_DEVICE_PROPERTY_SERIAL_NUMBER"].get<std::string>();
    std::cout << "Device " << i << " serial #: " << serial << std::endl;
    int32_t camId = -1;
    for (size_t j=0; j<rig.cams_.size(); ++j) {
      if (rig.serials_[j].compare(serial) == 0) {
        camId = j;
        break;
      }
    }
    if (camId < 0) {
      std::cerr << "no matching camera found in calibration!" << std::endl;
    } else {
      std::cout << "matching camera in config: " << camId << " " 
        << rig.config_[camId]["camera"]["description"].template get<std::string>()
        << std::endl;
    }
    rgbStream2cam.push_back(camId); // rgb
    dStream2cam.push_back(camId+1); // ir/depth
    rgbdStream2cam.push_back(camId/2); // rgbd
  }
  return true;
}

}
