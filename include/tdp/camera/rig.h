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
#include <tdp/gui/gui_base.hpp>
#include <tdp/manifold/SE3.h>
#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/preproc/normals.h>
#include <tdp/config.h>

namespace tdp {

template <class Cam>
struct Rig {

  ~Rig() {
//    for (size_t i=0; i<depthScales_.size(); ++i) {
//      delete[] depthScales_[i].ptr_;
//    }
  }

  bool ParseTransformation(const pangolin::json::value& jsT, SE3f& T) {
    pangolin::json::value t_json = jsT["t_xyz"];
    Eigen::Vector3f t(
        t_json[0].get<double>(),
        t_json[1].get<double>(),
        t_json[2].get<double>());
    Eigen::Matrix3f R;
    if (jsT.contains("q_wxyz")) {
      pangolin::json::value q_json = jsT["q_wxyz"];
      Eigen::Quaternionf q(q_json[0].get<double>(),
          q_json[1].get<double>(),
          q_json[2].get<double>(),
          q_json[3].get<double>());
      R = q.toRotationMatrix();
    } else if (jsT.contains("R_3x3")) {
      pangolin::json::value R_json = jsT["R_3x3"];
      R << R_json[0].get<double>(), R_json[1].get<double>(), 
           R_json[2].get<double>(), 
           R_json[3].get<double>(), R_json[4].get<double>(), 
           R_json[5].get<double>(), 
           R_json[6].get<double>(), R_json[7].get<double>(), 
           R_json[8].get<double>();
    } else {
      return false;
    }
    T = SE3f(R, t);
    return true;
  }

  bool FromFile(std::string pathToConfig, bool verbose) {
    pangolin::json::value file_json(pangolin::json::object_type,true); 
    std::ifstream f(pathToConfig);
    if (f.is_open()) {
      std::string err = pangolin::json::parse(file_json,f);
      if (!err.empty()) {
//        std::cout << file_json.serialize(true) << std::endl;
        if (file_json.size() > 0) {
          std::cout << "found " << file_json.size() << " elements" << std::endl ;
          cuDepthScales_.reserve(file_json.size());
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
                std::cout << file_json[i]["camera"].serialize(true) << std::endl;
              }
              cams_.push_back(cam);
              if (file_json[i]["camera"].contains("depthScale")) {
                std::string path = CONFIG_DIR+file_json[i]["camera"]["depthScale"].get<std::string>();
                depthScalePaths_.push_back(path);
                if (pangolin::FileExists(path)) {
                  pangolin::TypedImage scale8bit = pangolin::LoadImage(path);
                  size_t w = scale8bit.w/4;
                  size_t h = scale8bit.h;
                  std::cout << "w x h: " << w << "x" << h << std::endl;
                  Image<float> scaleWrap(w,h,w*sizeof(float),
                      (float*)scale8bit.ptr);
                  cuDepthScales_.emplace_back(w,h);
                  cuDepthScales_[cuDepthScales_.size()-1].CopyFrom(scaleWrap,
                      cudaMemcpyHostToDevice);
                  std::cout << "found and loaded depth scale file"
                    << " " 
                    <<  cuDepthScales_[cuDepthScales_.size()-1].ptr_ << std::endl;
                }
              }
              if (file_json[i]["camera"].contains("depthScaleVsDepthModel")) {
                scaleVsDepths_.push_back(Eigen::Vector2f(
                  file_json[i]["camera"]["depthScaleVsDepthModel"][0].get<double>(),
                  file_json[i]["camera"]["depthScaleVsDepthModel"][1].get<double>()));
              }
              if (file_json[i]["camera"].contains("depthSensorUniformScale")) {
                depthSensorUniformScale_.push_back(file_json[i]["camera"]["depthSensorUniformScale"].get<double>());
              }
              if (file_json[i]["camera"].contains("T_rc")) {
                SE3f T_rc;
                if (ParseTransformation(file_json[i]["camera"]["T_rc"],T_rc)) {
                  if (verbose) 
                    std::cout << "found T_rc" << std::endl << T_rc << std::endl;
                  T_rcs_.push_back(T_rc);
                }
              }
            } else if (file_json[i].contains("imu")) {
              std::cout << "found IMU: " << file_json[i]["imu"]["type"].get<std::string>() << std::endl;
              if (file_json[i]["imu"].contains("T_ri")) {
                SE3f T_ri;
                if (ParseTransformation(file_json[i]["imu"]["T_ri"],T_ri)) {
                  if (verbose) 
                    std::cout << "found T_ri" << std::endl << T_ri << std::endl;
                  T_ris_.push_back(T_ri);
                }
              }
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

  bool CorrespondOpenniStreams2Cams(
    const std::vector<pangolin::VideoInterface*>& streams);

  void CollectRGB(const GuiBase& gui,
    Image<Vector3bda>& rgb, cudaMemcpyKind type) ;

  void CollectD(const GuiBase& gui,
    float dMin, float dMax, Image<uint16_t>& cuDraw,
    Image<float>& cuD, int64_t& t_host_us_d) ;

  void ComputePc(Image<float>& cuD, bool useRgbCamParasForDepth, 
    Image<Vector3fda>& cuPc);

  template<int LEVELS>
  void ComputePc(Image<float>& cuD, bool useRgbCamParasForDepth, 
      Pyramid<Vector3fda,LEVELS>& cuPyrPc);

  void ComputeNormals(Image<float>& cuD, bool useRgbCamParasForDepth, 
    Image<Vector3fda>& cuN);

  template<int LEVELS>
  void ComputeNormals(Image<float>& cuD, bool useRgbCamParasForDepth, 
      Pyramid<Vector3fda,LEVELS>& cuPyrN);

  void AddToTSDF(const Image<float>& cuD, const SE3f& T_mr,
    bool useRgbCamParasForDepth, 
    const Vector3fda& grid0, const Vector3fda& dGrid,
    float tsdfMu, float tsdfWMax,
    Volume<TSDFval>& cuTSDF);

  template<int LEVELS>
  void RayTraceTSDF(
      const Volume<TSDFval>& cuTSDF const SE3f& T_mr,
      bool useRgbCamParasForDepth, 
      const Vector3fda& grid0, const Vector3fda& dGrid,
      float tsdfMu, float tsdfWThr,
      Pyramid<Vector3fda,LEVELS>& cuPyrPc,
      Pyramid<Vector3fda,LEVELS>& cuPyrN);

  size_t NumStreams() { return rgbdStream2cam_.size(); }
  size_t NumCams() { return rgbdStream2cam_.size()/2; }

  void Render3D(const SE3f& T_mr, float scale=1.);

  template <typename T>
  Image<T> GetStreamRoI(const Image<T>& I, size_t streamId) {
    return I.GetRoi(0, rgbdStream2cam_[streamId]*hSingle, wSingle, hSingle);
  };

  // imu to rig transformations
  std::vector<SE3f> T_ris_; 
  // camera to rig transformations
  std::vector<SE3f> T_rcs_; 
  // cameras
  std::vector<Cam> cams_;

  std::vector<float> depthSensorUniformScale_;
  // depth scale calibration images
  std::vector<std::string> depthScalePaths_;
//  std::vector<Image<float>> depthScales_;
  std::vector<ManagedDeviceImage<float>> cuDepthScales_;
  // depth scale scaling model as a function of depth
  eigen_vector<Eigen::Vector2f> scaleVsDepths_;

  std::vector<int32_t> rgbStream2cam_;
  std::vector<int32_t> dStream2cam_;
  std::vector<int32_t> rgbdStream2cam_;

  size_t wSingle;
  size_t hSingle;

  // camera serial IDs
  std::vector<std::string> serials_;
  // raw properties
  pangolin::json::value config_;
};

/// Uses serial number in openni device props and rig to find
/// correspondences.
template<class CamT>
bool Rig<CamT>::CorrespondOpenniStreams2Cams(
    const std::vector<pangolin::VideoInterface*>& streams) {

  rgbStream2cam_.clear();
  dStream2cam_.clear();
  rgbdStream2cam_.clear();
  
  pangolin::json::value devProps = pangolin::GetVideoDeviceProperties(streams[0]);

  std::string devType = "";
  if (devProps.contains("openni")) {
    devType = "openni"; 
  } else if(devProps.contains("realsense"))  {
    devType = "realsense"; 
  } else {
    return false;
  }
  pangolin::json::value jsDevices = devProps[devType]["devices"];

  for (size_t i=0; i<jsDevices.size(); ++i) {
    std::string serial;
    if (jsDevices[i].contains("ONI_DEVICE_PROPERTY_SERIAL_NUMBER")) {
      serial = jsDevices[i]["ONI_DEVICE_PROPERTY_SERIAL_NUMBER"].get<std::string>();
    } else if (jsDevices[i].contains("serial_number")) {
      serial = jsDevices[i]["serial_number"].get<std::string>();
    }
    std::cout << "Device " << i << " serial #: " << serial << std::endl;
    int32_t camId = -1;
    for (size_t j=0; j<cams_.size(); ++j) {
      if (serials_[j].compare(serial) == 0) {
        camId = j;
        break;
      }
    }
    if (camId < 0) {
      std::cerr << "no matching camera found in calibration!" << std::endl;
    } else {
      std::cout << "matching camera in config: " << camId << " " 
        << config_[camId]["camera"]["description"].template get<std::string>()
        << std::endl;
    }
    rgbStream2cam_.push_back(camId); // rgb
    dStream2cam_.push_back(camId+1); // ir/depth
    rgbdStream2cam_.push_back(camId/2); // rgbd
  }
  return true;
}

template<class CamT>
void Rig<CamT>::CollectRGB(const GuiBase& gui,
    Image<Vector3bda>& rgb, cudaMemcpyKind type) {
  for (size_t sId=0; sId < rgbdStream2cam_.size(); sId++) {
    Image<Vector3bda> rgbStream;
    if (!gui.ImageRGB(rgbStream, sId)) continue;
    // TODO: this is a bit hackie; should get the w and h somehow else
    // beforehand
    int32_t cId = rgbdStream2cam_[sId]; 
    wSingle = rgbStream.w_+rgbStream.w_%64;
    hSingle = rgbStream.h_+rgbStream.h_%64;
    Image<Vector3bda> rgb_i = rgb.GetRoi(0,cId*hSingle, wSingle, hSingle);
    rgb_i.CopyFrom(rgbStream,type);
  }
}

template<class CamT>
void Rig<CamT>::CollectD(const GuiBase& gui,
    float dMin, float dMax, Image<uint16_t>& cuDraw,
    Image<float>& cuD, int64_t& t_host_us_d) {

//  tdp::ManagedDeviceImage<float> cuScale(wSingle, hSingle);
  int32_t numStreams = 0;
  t_host_us_d = 0;
  for (size_t sId=0; sId < rgbdStream2cam_.size(); sId++) {
    tdp::Image<uint16_t> dStream;
    int64_t t_host_us_di = 0;
    if (!gui.ImageD(dStream, sId, &t_host_us_di)) continue;
    // TODO: this is a bit hackie; should get the w and h somehow else
    // beforehand
    t_host_us_d += t_host_us_di;
    numStreams ++;
    int32_t cId = rgbdStream2cam_[sId]; 
    wSingle = dStream.w_+dStream.w_%64;
    hSingle = dStream.h_+dStream.h_%64;
    tdp::Image<uint16_t> cuDraw_i = cuDraw.GetRoi(0,cId*hSingle,
        wSingle, hSingle);
    cuDraw_i.CopyFrom(dStream,cudaMemcpyHostToDevice);
    // convert depth image from uint16_t to float [m]
    tdp::Image<float> cuD_i = cuD.GetRoi(0, cId*hSingle, wSingle, hSingle);
    if (cuDepthScales_.size() > cId) {
      float a = scaleVsDepths_[cId](0);
      float b = scaleVsDepths_[cId](1);
      tdp::ConvertDepthGpu(cuDraw_i, cuD_i, cuDepthScales_[cId], 
          a, b, dMin, dMax);
    } else if (depthSensorUniformScale_.size() > cId) {
      tdp::ConvertDepthGpu(cuDraw_i, cuD_i, depthSensorUniformScale_[cId], dMin, dMax);
    } else {
       std::cout << "Warning no scale information found" << std::endl;
    }
  }
  t_host_us_d /= numStreams;  
}

template<class CamT>
void Rig<CamT>::ComputeNormals(Image<float>& cuD,
    bool useRgbCamParasForDepth, 
    Image<Vector3fda>& cuN) {
  for (size_t sId=0; sId < dStream2cam_.size(); sId++) {
    int32_t cId;
    if (useRgbCamParasForDepth) {
      cId = rgbStream2cam_[sId]; 
    } else {
      cId = dStream2cam_[sId]; 
    }
    CamT cam = cams_[cId];
    tdp::SE3f T_rc = T_rcs_[cId];

    tdp::Image<tdp::Vector3fda> cuN_i = cuN.GetRoi(0,
        rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
    tdp::Image<float> cuD_i = cuD.GetRoi(0,
        rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
    // compute normals from depth in rig coordinate system
    tdp::Depth2Normals(cuD_i, cam, T_rc.rotation(), cuN_i);
  }
}

template<class CamT>
template<int LEVELS>
void Rig<CamT>::ComputeNormals<LEVELS>(Image<float>& cuD, 
    bool useRgbCamParasForDepth, 
    Pyramid<Vector3fda,LEVELS>& cuPyrN) {
  Image<Vector3fda> cuPc = cuPyrN.GetImage(0);
  ComputeNormals(cuD, useRgbCamParasForDepth, cuN);
  tdp::CompleteNormalPyramid<tdp::Vector3fda,LEVELS>(cuPyrN,cudaMemcpyDeviceToDevice);
}

template<class CamT>
void Rig<CamT>::ComputePc(Image<float>& cuD, 
    bool useRgbCamParasForDepth, 
    Image<Vector3fda>& cuPc) {
  for (size_t sId=0; sId < dStream2cam_.size(); sId++) {
    int32_t cId;
    if (useRgbCamParasForDepth) {
      cId = rgbStream2cam_[sId]; 
    } else {
      cId = dStream2cam_[sId]; 
    }
    CamT cam = cams_[cId];
    tdp::SE3f T_rc = T_rcs_[cId];

    tdp::Image<tdp::Vector3fda> cuPc_i = cuPc.GetRoi(0,
        rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
    tdp::Image<float> cuD_i = cuD.GetRoi(0,
        rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);

    // compute point cloud from depth in rig coordinate system
    tdp::Depth2PCGpu(cuD_i, cam, T_rc, cuPc_i);
  }
}

template<class CamT>
template<int LEVELS>
void Rig<CamT>::ComputePc<LEVELS>(Image<float>& cuD, 
    bool useRgbCamParasForDepth, 
    Pyramid<Vector3fda,LEVELS>& cuPyrPc) {
  Image<Vector3fda> cuPc = cuPyrPc.GetImage(0);
  ComputePc(cuD, useRgbCamParasForDepth, cuPc);
  tdp::CompletePyramid<tdp::Vector3fda,LEVELS>(cuPyrPc,cudaMemcpyDeviceToDevice);
}


template<class CamT>
void Rig<CamT>::AddToTSDF(const Image<float>& cuD, 
    const SE3f& T_mr,
    bool useRgbCamParasForDepth, 
    const Vector3fda& grid0,
    const Vector3fda& dGrid,
    float tsdfMu,
    float tsdfWMax,
    Volume<TSDFval>& cuTSDF) {
  for (size_t sId=0; sId < dStream2cam_.size(); sId++) {
    int32_t cId;
    if (useRgbCamParasForDepth) {
      cId = rgbStream2cam_[sId]; 
    } else {
      cId = dStream2cam_[sId]; 
    }
    CamT cam = cams_[cId];
    tdp::SE3f T_rc = T_rcs_[cId];
    tdp::SE3f T_mo = T_mr*T_rc;
    tdp::Image<float> cuD_i(wSingle, hSingle,
        cuD.ptr_+rgbdStream2cam_[sId]*wSingle*hSingle);
    AddToTSDF(cuTSDF, cuD_i, T_mo, cam, grid0, dGrid, tsdfMu, tsdfWMax); 
  }
}

template<class CamT>
template<int LEVELS>
void Rig<CamT>::RayTraceTSDF<LEVELS>(
    const Volume<TSDFval>& cuTSDF
    const SE3f& T_mr,
    bool useRgbCamParasForDepth, 
    const Vector3fda& grid0,
    const Vector3fda& dGrid,
    float tsdfMu,
    float tsdfWThr,
    Pyramid<Vector3fda,LEVELS>& cuPyrPc,
    Pyramid<Vector3fda,LEVELS>& cuPyrN) {
  tdp::Image<tdp::Vector3fda> cuNEst = cuPyrN.GetImage(0);
  tdp::Image<tdp::Vector3fda> cuPcEst = cuPyrPc.GetImage(0);
  for (size_t sId=0; sId < dStream2cam_.size(); sId++) {
    int32_t cId;
    if (useRgbCamParasForDepth) {
      cId = rgbStream2cam_[sId]; 
    } else {
      cId = dStream2cam_[sId]; 
    }
    CamT cam = cams_[cId];
    tdp::SE3f T_rc = T_rcs_[cId];
    tdp::SE3f T_mo = T_mr*T_rc;

    tdp::Image<tdp::Vector3fda> cuNEst_i = cuNEst.GetRoi(0,
        rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
    tdp::Image<tdp::Vector3fda> cuPcEst_i = cuPcEst.GetRoi(0,
        rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);

    // ray trace the TSDF to get pc and normals in model cosy
    RayTraceTSDF(cuTSDF, cuPcEst_i, 
        cuNEst_i, T_mo, cam, grid0, dGrid, tsdfMu, tsdfWThr); 
  }
  // just complete the surface normals obtained from the TSDF
  tdp::CompletePyramid<tdp::Vector3fda,LEVELS>(cuPyrPc,cudaMemcpyDeviceToDevice);
  tdp::CompleteNormalPyramid<LEVELS>(cuPyrN,cudaMemcpyDeviceToDevice);
}

template<class CamT>
void Rig<CamT>::Render3D(
    const SE3f& T_mr,
    float scale) {

  for (size_t sId=0; sId < dStream2cam_.size(); sId++) {
    int32_t cId;
    if (useRgbCamParasForDepth) {
      cId = rgbStream2cam_[sId]; 
    } else {
      cId = dStream2cam_[sId]; 
    }
    CamT cam = cams_[cId];
    tdp::SE3f T_rc = T_rcs_[cId];
    tdp::SE3f T_mo = T_mr*T_rc;
    
    pangolin::glDrawFrustrum(cam.GetKinv(), wSingle, hSingle,
        T_mo.matrix(), scale);
    pangolin::glDrawAxis(T_mo.matrix(), scale);

  }
  pangolin::glDrawAxis(T_mr.matrix(), scale);
}

}
