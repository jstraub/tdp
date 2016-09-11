
#pragma once

#include <pangolin/gl/gldraw.h>
#include <aruco/aruco.h>
#include <opencv2/highgui/highgui.hpp>

#include <tdp/manifold/SE3.h>
#include <tdp/data/image.h>
#include <tdp/camera/camera.h>

namespace tdp {

/// Wrapper around aruco Marker
struct Marker {
  SE3f T_cm;

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

  std::vector<Marker> detect(Image<Vector3bda> rgb, const Cameraf& cam);
  std::vector<Marker> detect(Image<uint8_t> grey, const Cameraf& cam);

 private:
  float markerSideLength_;
  aruco::MarkerDetector detector_;
};




}

