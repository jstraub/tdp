/*
 * This file is part of the Calibu Project.
 * https://github.com/gwu-robotics/Calibu
 *
 * Copyright (C) 2013 George Washington University,
 *                    Hauke Strasdat,
 *                    Steven Lovegrove
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This code is heavily based on Calibu/include/calibu/pose/Pnp.h and
 * the respective Pnp.cpp. The modifications are to use the tdp::Camera as
 * well as tdp::SE3 classes.
 *
 * Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <vector>
#include <tdp/camera/camera.h>
#include <tdp/manifold/SE3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace tdp {

std::vector<int> PnPRansac(
    const Camera<float>& cam,
    const std::vector<Eigen::Vector2d,
      Eigen::aligned_allocator<Eigen::Vector2d> >& pts2D,
    const std::vector<Eigen::Vector3d, 
      Eigen::aligned_allocator<Eigen::Vector3d> >& pts3D,
    const std::vector<int>& candidate_map,
    int robust_3pt_its,
    float robust_3pt_tol,
    SE3f& T) {

  std::vector<int> inlier_map(candidate_map.size(), -1);
  std::vector<cv::Point3f> cv_obj;
  std::vector<cv::Point2f> cv_img;
  std::vector<int> idx_vec;
  cv::Mat cv_coeff;
  cv::Mat cv_rot(3,1,CV_64F);
  cv::Mat cv_trans(3,1,CV_64F);
  cv::Mat cv_K(3,3,CV_64F);

  //  cv::eigen2cv(cam.K(), cv_K);
  cv::setIdentity(cv_K);

  for (size_t i=0; i<pts2D.size(); ++i)
  {
    int ideal_point_id = candidate_map[i];
    if (ideal_point_id >= 0)
    {
      const Eigen::Vector3d img_center_pts = cam.Unproject(pts2D[i](0),
          pts2D[i](1),1.).cast<double>();
      Eigen::Vector2d center(img_center_pts[0]/img_center_pts[2],
          img_center_pts[1]/img_center_pts[2]);
      // const Eigen::Vector2d center = cam.Unmap(pts2D[i]);
      const Eigen::Vector3d & c3d = pts3D[ideal_point_id];
      cv_img.push_back(cv::Point2f(center(0), center(1)));
      cv_obj.push_back(cv::Point3f(c3d(0), c3d(1), c3d(2)));
      idx_vec.push_back(i);
    }
  }

  std::vector<int> cv_inliers;

  if(cv_img.size() < 4)
    return cv_inliers;

  if(robust_3pt_its > 0) {
    cv::solvePnPRansac(cv_obj, cv_img, cv_K, cv_coeff, cv_rot, cv_trans,
        false, robust_3pt_its, robust_3pt_tol / cam.GetK()(0,0), 60, cv_inliers);
  }else{
    cv::solvePnP(cv_obj, cv_img, cv_K, cv_coeff, cv_rot, cv_trans, false);
  }

  Eigen::Vector3d rot, trans;
  cv::cv2eigen(cv_rot, rot);
  cv::cv2eigen(cv_trans, trans);

  if(std::isnan(rot[0]) || std::isnan(rot[1]) || std::isnan(rot[2]))
    return inlier_map;

  for (size_t i=0; i<cv_inliers.size(); ++i) {
    int idx = cv_inliers[i];
    inlier_map[idx_vec.at(idx)] = candidate_map[idx_vec.at(idx)];
  }

  T = SE3f(SO3f::Exp_(rot.cast<float>()).matrix(), trans.cast<float>());
  return inlier_map;
}

}
