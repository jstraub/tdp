#include <tdp/marker/aruco.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace tdp {

//std::vector<Marker> ArucoDetector::detect(Image<Vector3bda> rgb, const
//    Cameraf& cam) {
//  cv::Mat cvRgb(rgb.h_, rgb.w_, CV_8UC3, rgb.ptr_);
//  cv::Mat cvGrey(rgb.h_, rgb.w_, CV_8UC1);
//  cv::cvtColor(cvRgb, cvGrey, CV_BGR2GRAY);
//  Image<uint8_t> grey(rgb.w_, rgb.h_, cvGrey.data);
//  return detect(grey, cam);
//}
//
//std::vector<Marker> ArucoDetector::detect(Image<uint8_t> grey, const
//    Cameraf& cam) {
//  cv::Mat cvGrey(grey.h_, grey.w_, CV_8UC1, grey.ptr_);
//  Eigen::Matrix<float,3,3,Eigen::RowMajor> K_ = cam.GetK(); cv::Mat K(3,3, CV_32F, K_.data());
//  // TODO: distortion
//  Eigen::Vector4f dist_(0,0,0,0);
//  cv::Mat distortion (4,1, CV_32F, dist_.data());
//  aruco::CameraParameters CP(K,distortion,cv::Size(grey.h_,grey.w_));
//  std::vector<aruco::Marker> arucoMarkers = detector_.detect(cvGrey);
//  //, CP, markerSideLength_);
//  std::vector<Marker> markers(arucoMarkers.size());
//
//  for (size_t i=0; i<arucoMarkers.size(); ++i) {
//    arucoMarkers[i].calculateExtrinsics(markerSideLength_, CP, true);
//    Eigen::Matrix<float,6,1> x; 
//    x.topRows(3) = Eigen::Map<Eigen::Vector3f>(
//        (float*)arucoMarkers[i].Rvec.data);
//    x.bottomRows(3) = Eigen::Map<Eigen::Vector3f>(
//        (float*)arucoMarkers[i].Tvec.data);
//    markers[i].T_cm = tdp::SE3f(tdp::SE3f::Exp_(x));
//    markers[i].aruco_ = arucoMarkers[i];
//    markers[i].id = arucoMarkers[i].id;
//  }
//
//  return markers;
//}

}
