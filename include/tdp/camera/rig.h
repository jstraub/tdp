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

  bool FromFile(std::string pathToConfig) {
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
              Cam cam;
              if (cam.FromJson(file_json[i]["camera"])) {
                std::cout << "found camera" << std::endl 
                  << file_json[i].serialize(true)
                  << std::endl;
              }
              cams_.push_back(cam);
              // TODO: poses
              SE3f T_rc;
              if (file_json[i]["camera"].contains("T_rc")) {
                pangolin::json::value q_json = file_json[i]["camera"]["T_rc"]["q_wxyz"];
                pangolin::json::value t_json = file_json[i]["camera"]["T_rc"]["t_xyz"];
                Eigen::Quaternionf q(q_json[0].get<double>(),
                    q_json[1].get<double>(),
                    q_json[2].get<double>(),
                    q_json[3].get<double>());
                Eigen::Vector3f t(
                    t_json[0].get<double>(),
                    t_json[1].get<double>(),
                    t_json[2].get<double>());
                Eigen::Matrix3f R_rc = q.toRotationMatrix();
                T_rc = SE3f(R_rc, t);
                std::cout << "found T_rc" << std::endl 
                  << q.coeffs().transpose() << std::endl
                  << R_rc << std::endl
                  << T_rc << std::endl;
              }
              T_rcs_.push_back(T_rc);
            }
          }
        }
      } else {
        std::cerr << "error reading json file: " << err << std::endl;
      }
    }
  }
  // camera to rig transformations
  std::vector<SE3f> T_rcs_; 
  // cameras
  std::vector<Cam> cams_;
};

}
