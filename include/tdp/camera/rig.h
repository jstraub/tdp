#pragma once

#include <vector>
#include <iostream>
#include <fstream>

#include <tdp/manifold/SE3.h>
#include <tdp/camera/camera.h>
#include <pangolin/utils/picojson.h>

namespace tdp {
template <class Cam>
struct Rig {

  bool FromFile(std::string pathToConfig, bool verbose) {
    pangolin::json::value file_json(pangolin::json::object_type,true); 
    std::ifstream f(pathToConfig);
    if (f.is_open()) {
      std::string err = pangolin::json::parse(file_json,f);
      if (!err.empty()) {
        std::cout << file_json.serialize(true) << std::endl;
        if (file_json.size() > 0) {
          std::cout << "found " << file_json.size() << " elements" << std::endl ;
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
  // camera serial IDs
  std::vector<std::string> serials_;
  // raw properties
  pangolin::json::value config_;
};

}
