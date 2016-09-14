
#pragma once

#include <pangolin/gl/gldraw.h>
#include <aruco/aruco.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tdp/manifold/SE3.h>
#include <tdp/data/image.h>
#include <tdp/camera/camera_base.h>
#include <tdp/eigen/dense.h>

namespace tdp {

/// Wrapper around aruco Marker
struct Marker {
  SE3f T_cm;
  size_t id;
  aruco::Marker aruco_;

  void drawToImage(Image<Vector3bda>& rgb, Vector3bda color, int lineWidth) {
    cv::Mat cvRgb(rgb.h_, rgb.w_, CV_8UC3, rgb.ptr_);
    aruco_.draw(cvRgb, cv::Scalar(color(0),color(1),color(2)), lineWidth);
  }

  void draw3D() {
    pangolin::glSetFrameOfReference(T_cm.matrix());
    pangolin::glDrawAxis(0.1f);
    pangolin::glUnsetFrameOfReference();
  }

};

/// Wrapper around aruco MarkerDetector 
class ArucoDetector {
 public:
  ArucoDetector(float markerSideLength) : markerSideLength_(markerSideLength)
  {
    detector_.setDictionary(aruco::Dictionary::ARUCO_MIP_36h12);
    detector_.setThresholdParams(7, 7); 
    detector_.setThresholdParamRange(2, 0); 
  }
  ~ArucoDetector() {}

  template<int D, typename Derived>
  std::vector<Marker> detect(Image<Vector3bda> rgb, 
      const CameraBase<float,D,Derived>& cam);

  template<int D, typename Derived>
  std::vector<Marker> detect(Image<uint8_t> grey, 
      const CameraBase<float,D,Derived>& cam);

 private:
  float markerSideLength_;
  aruco::MarkerDetector detector_;
};

template<int D, typename Derived>
std::vector<Marker> ArucoDetector::detect(Image<Vector3bda> rgb, 
    const CameraBase<float,D,Derived>& cam) {
  cv::Mat cvRgb(rgb.h_, rgb.w_, CV_8UC3, rgb.ptr_);
  cv::Mat cvGrey(rgb.h_, rgb.w_, CV_8UC1);
  cv::cvtColor(cvRgb, cvGrey, CV_BGR2GRAY);
  Image<uint8_t> grey(rgb.w_, rgb.h_, cvGrey.data);
  return detect(grey, cam);
}

template<int D, typename Derived>
std::vector<Marker> ArucoDetector::detect(Image<uint8_t> grey, 
    const CameraBase<float,D,Derived>& cam) {
  cv::Mat cvGrey(grey.h_, grey.w_, CV_8UC1, grey.ptr_);
  Eigen::Matrix<float,3,3,Eigen::RowMajor> K_ = cam.GetK(); 
  cv::Mat K(3,3, CV_32F, K_.data());
  // TODO: distortion
  Eigen::Vector4f dist_(0,0,0,0);
  cv::Mat distortion (4,1, CV_32F, dist_.data());
  aruco::CameraParameters CP(K,distortion,cv::Size(grey.h_,grey.w_));
  std::vector<aruco::Marker> arucoMarkers = detector_.detect(cvGrey);
  //, CP, markerSideLength_);
  std::vector<Marker> markers(arucoMarkers.size());

  for (size_t i=0; i<arucoMarkers.size(); ++i) {
    arucoMarkers[i].calculateExtrinsics(markerSideLength_, CP, true);
    Eigen::Matrix<float,6,1> x; 
    x.topRows(3) = Eigen::Map<Eigen::Vector3f>(
        (float*)arucoMarkers[i].Rvec.data);
    x.bottomRows(3) = Eigen::Map<Eigen::Vector3f>(
        (float*)arucoMarkers[i].Tvec.data);
    markers[i].T_cm = tdp::SE3f(tdp::SE3f::Exp_(x));
    markers[i].aruco_ = arucoMarkers[i];
    markers[i].id = arucoMarkers[i].id;
  }

  return markers;
}


}

