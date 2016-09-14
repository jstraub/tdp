/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <vector>
#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image_io.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#include <tdp/utils/Stopwatch.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include "gui.hpp"
#include <tdp/camera/rig.h>
#include <tdp/marker/aruco.h>
#include <tdp/manifold/SE3.h>
#include <pangolin/video/drivers/openni2.h>

typedef tdp::CameraPoly3<float> CameraT;

void VideoViewer(const std::string& input_uri, 
    const std::string& configPath,
    const std::string& output_uri)
{
  std::cout << " -!!- this application works only with openni2 devices (tested with Xtion PROs) -!!- " << std::endl;

  // Read rig file
  tdp::Rig<CameraT> rig;
  if (!rig.FromFile(configPath, true)) return;

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return;
  }
  std::vector<int32_t> rgbStream2cam;
  std::vector<int32_t> dStream2cam;
  std::vector<int32_t> rgbdStream2cam;
  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  tdp::CorrespondOpenniStreams2Cams(streams,rig,rgbStream2cam,
      dStream2cam, rgbdStream2cam);

  tdp::GUInoViews gui(1200,800,video);

  size_t wSingle = video.Streams()[0].Width();
  size_t hSingle = video.Streams()[0].Height();
  size_t w = wSingle;
  size_t h = 3*hSingle;
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  w += w%64;
  h += h%64;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);

  tdp::QuickView viewRgb(w,h);
  gui.container().AddDisplay(viewRgb);
  tdp::QuickView viewD(w,h);
  gui.container().AddDisplay(viewD);
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPc(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensor1Scale("ui.depth1 scale",1e-3,8e-4,1e-3);
  pangolin::Var<float> depthSensor2Scale("ui.depth2 scale",1e-3,8e-4,1e-3);
  pangolin::Var<float> depthSensor3Scale("ui.depth3 scale",1e-3,8e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> doRigPoseCalib("ui.Rig Pose Calib", false, true);
  pangolin::Var<bool> updateCalib("ui.update Calib", false, false);

  pangolin::Var<float> cam1fu("ui.cam1 fu",rig.cams_[1].params_(0),500,600);
  pangolin::Var<float> cam1fv("ui.cam1 fv",rig.cams_[1].params_(1),500,600);
  pangolin::Var<float> cam3fu("ui.cam3 fu",rig.cams_[3].params_(0),500,600);
  pangolin::Var<float> cam3fv("ui.cam3 fv",rig.cams_[3].params_(1),500,600);
  pangolin::Var<float> cam5fu("ui.cam5 fu",rig.cams_[5].params_(0),500,600);
  pangolin::Var<float> cam5fv("ui.cam5 fv",rig.cams_[5].params_(1),500,600);

  pangolin::Var<float> cam3tx("ui.cam3 tx",rig.T_rcs_[3].translation()(0),0,0.1);
  pangolin::Var<float> cam3ty("ui.cam3 ty",rig.T_rcs_[3].translation()(1),0,0.1);
  pangolin::Var<float> cam3tz("ui.cam3 tz",rig.T_rcs_[3].translation()(2),0,0.1);

  pangolin::RegisterKeyPressCallback('c', [&](){
      for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
      int cId = rgbdStream2cam[sId];
      std::stringstream ss;
      ss << "capture_cam" << cId << ".png";
      try{
      pangolin::SaveImage(
        gui.images[gui.iRGB[sId]], gui.video.Streams()[gui.iRGB[sId]].PixFormat(),
        pangolin::MakeUniqueFilename(ss.str())
        );
      }catch(std::exception e){
      pango_print_error("Unable to save frame: %s\n", e.what());
      }
      }
      } );

  tdp::ArucoDetector detector(0.158);
  // observed transformations from upper camera to middle camera (rig
  // cosy)
  std::vector<tdp::SE3f> T_rcu; 
  // observed transformations from lower camera to middle camera (rig
  // cosy)
  std::vector<tdp::SE3f> T_rcl;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (cam1fu.GuiChanged()) rig.cams_[1].params_(0) = cam1fu;
    if (cam1fv.GuiChanged()) rig.cams_[1].params_(1) = cam1fv;
    if (cam3fu.GuiChanged()) rig.cams_[3].params_(0) = cam3fu;
    if (cam3fv.GuiChanged()) rig.cams_[3].params_(1) = cam3fv;
    if (cam5fu.GuiChanged()) rig.cams_[5].params_(0) = cam5fu;
    if (cam5fv.GuiChanged()) rig.cams_[5].params_(1) = cam5fv;

    if (cam3tx.GuiChanged()) rig.T_rcs_[3].matrix()(0,3) = cam3tx;
    if (cam3ty.GuiChanged()) rig.T_rcs_[3].matrix()(1,3) = cam3ty;
    if (cam3tz.GuiChanged()) rig.T_rcs_[3].matrix()(2,3) = cam3tz;

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    TICK("rgb collection");
    // get rgb image
    for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
      tdp::Image<tdp::Vector3bda> rgbStream;
      if (!gui.ImageRGB(rgbStream, sId)) continue;
      int32_t cId = rgbdStream2cam[sId]; 
      // Get ROI
      tdp::Image<tdp::Vector3bda> rgb_i(wSingle, hSingle,
          rgb.ptr_+cId*rgbStream.Area());
      rgb_i.CopyFrom(rgbStream,cudaMemcpyHostToHost);
    }
    TOCK("rgb collection");
    TICK("depth collection");
    // get depth image
    for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
      tdp::Image<uint16_t> dStream;
      if (!gui.ImageD(dStream, sId)) continue;
      int32_t cId = rgbdStream2cam[sId]; 
      //std::cout << sId << " " << cId << std::endl;
      // Get ROI
      tdp::Image<uint16_t> cuDraw_i(wSingle, hSingle,
          cuDraw.ptr_+cId*dStream.Area());
      cuDraw_i.CopyFrom(dStream,cudaMemcpyHostToDevice);
      // convert depth image from uint16_t to float [m]
      tdp::Image<float> cuD_i(wSingle, hSingle,
          cuD.ptr_+cId*dStream.Area());
      float depthSensorScale = depthSensor1Scale;
      if (cId==1) depthSensorScale = depthSensor2Scale;
      if (cId==2) depthSensorScale = depthSensor3Scale;
      tdp::ConvertDepthGpu(cuDraw_i, cuD_i, depthSensorScale, dMin, dMax);
    }
    TOCK("depth collection");
    TICK("pc and normals");
    // convert depth image from uint16_t to float [m]
    //tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    // compute point cloud (on CPU)
    for (size_t sId=0; sId < dStream2cam.size(); sId++) {
      int32_t cId = dStream2cam[sId]; 
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      std::cout << cId << ": " << cam.params_.transpose() << std::endl
        << T_rc << std::endl;

      tdp::Image<tdp::Vector3fda> cuN_i(wSingle, hSingle,
          cuN.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);
      tdp::Image<tdp::Vector3fda> cuPc_i(wSingle, hSingle,
          cuPc.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);
      tdp::Image<float> cuD_i(wSingle, hSingle,
          cuD.ptr_+rgbdStream2cam[sId]*wSingle*hSingle);

      // compute depth
      tdp::Depth2PCGpu(cuD_i,cam,T_rc,cuPc_i);
      // compute normals
      tdp::Depth2Normals(cuD_i, cam, T_rc.rotation(), cuN_i);
    }
    TOCK("pc and normals");

    std::vector<std::vector<tdp::Marker>> markersPerCam(rgbStream2cam.size());
    if (doRigPoseCalib) {
      for (size_t sId=0; sId < rgbStream2cam.size(); sId++) {
        tdp::Image<tdp::Vector3bda> rgbStream;
        if (!gui.ImageRGB(rgbStream, sId)) continue;
        int32_t cId = rgbStream2cam[sId]; 
        CameraT cam = rig.cams_[cId];
        tdp::Image<tdp::Vector3bda> rgb_i(wSingle, hSingle,
            rgb.ptr_+rgbdStream2cam[sId]*rgbStream.Area());
        TICK("aruco marker detect");
        std::vector<tdp::Marker> markers = detector.detect(rgb_i, cam);
        TOCK("aruco marker detect");
        markersPerCam[cId/2] = markers;
        for (size_t i=0; i<markers.size(); ++i) {
          markers[i].drawToImage(rgb_i, tdp::Vector3bda(0,0,255), 1);
        }
      }
      // add observations between rig and upper cam
      // upper = cId=0
      // rig/middle = cId=1
      for (size_t i=0; i<markersPerCam[0].size(); ++i) {
        for (size_t j=0; j<markersPerCam[1].size(); ++j) {
          if (markersPerCam[0][i].id == markersPerCam[1][j].id) {
            tdp::SE3f T_cum = markersPerCam[0][i].T_cm;
            tdp::SE3f T_crm = markersPerCam[1][j].T_cm;
            T_rcu.push_back(T_crm+T_cum.Inverse());
          }
        }
      }
      // add observations between rig and lower cam
      // lower = cId=2
      // rig/middle = cId=1
      for (size_t i=0; i<markersPerCam[2].size(); ++i) {
        for (size_t j=0; j<markersPerCam[1].size(); ++j) {
          if (markersPerCam[2][i].id == markersPerCam[1][j].id) {
            tdp::SE3f T_clm = markersPerCam[2][i].T_cm;
            tdp::SE3f T_crm = markersPerCam[1][j].T_cm;
            T_rcl.push_back(T_crm+T_clm.Inverse());
          }
        }
      }
      std::cout << " # obs upper: " << T_rcu.size()
        << " # obs lower: " << T_rcl.size() << std::endl;
      // update upper rig transformations
      Eigen::Matrix<float,6,1> xSum;
      if (T_rcu.size() > 0 && updateCalib) {
        for (size_t i=0; i<T_rcu.size(); ++i) {
          xSum += rig.T_rcs_[0].Log(T_rcu[i]);
          std::cout << "T_rcu obs: " << i << ": " << std::endl
            << T_rcu[i] << std::endl;
        }
        std::cout << xSum/float(T_rcu.size()) << std::endl;
        std::cout << rig.T_rcs_[0] << std::endl;
        rig.T_rcs_[0] = rig.T_rcs_[0].Exp(xSum/float(T_rcu.size())); 
        rig.T_rcs_[1] = rig.T_rcs_[0]; 
        std::cout << "current T_rcu: " << std::endl
          << rig.T_rcs_[0] << std::endl;
      }
      // update lower rig transformations
      if (T_rcl.size() > 0 && pangolin::Pushed(updateCalib)) {
        xSum.fill(0.);
        for (size_t i=0; i<T_rcl.size(); ++i) {
          xSum += rig.T_rcs_[4].Log(T_rcl[i]);
        }
        rig.T_rcs_[4] = rig.T_rcs_[4].Exp(xSum/float(T_rcl.size())); 
        rig.T_rcs_[5] = rig.T_rcs_[4];
      }
    }

    // convert normals to RGB image
    tdp::Normals2Image(cuN, cuN2D);
    // copy to CPU memory for vis
    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
    n2D.CopyFrom(cuN2D,cudaMemcpyDeviceToHost);
    pc.CopyFrom(cuPc,cudaMemcpyDeviceToHost);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    for (size_t i=0; i<rgbStream2cam.size(); ++i) {
      for (size_t j=0; j<markersPerCam[i].size(); ++j) {
        pangolin::glSetFrameOfReference((rig.T_rcs_[rgbStream2cam[i]]
              +markersPerCam[i][j].T_cm).matrix());
        pangolin::glDrawAxis(0.1f);
        pangolin::glUnsetFrameOfReference();
      }
    }

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    viewRgb.SetImage(rgb);
    viewD.SetImage(d);
    viewN2D.SetImage(n2D);

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // if we are recording
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    // finish this frame
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }
}


int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";

  if( argc > 1 ) {
    const std::string input_uri = std::string(argv[1]);
    const std::string configPath = (argc > 2) ? std::string(argv[2]) : "../config/surround3D_2016_09_11.json";
    try{
      VideoViewer(input_uri, configPath, dflt_output_uri);
    } catch (pangolin::VideoException e) {
      std::cout << e.what() << std::endl;
    }
  }else{
    const std::string input_uris[] = {
      "dc1394:[fps=30,dma=10,size=640x480,iso=400]//0",
      "convert:[fmt=RGB24]//v4l:///dev/video0",
      "convert:[fmt=RGB24]//v4l:///dev/video1",
      "openni:[img1=rgb]//",
      "test:[size=160x120,n=1,fmt=RGB24]//"
        ""
    };

    std::cout << "Usage  : VideoViewer [video-uri]" << std::endl << std::endl;
    std::cout << "Where video-uri describes a stream or file resource, e.g." << std::endl;
    std::cout << "\tfile:[realtime=1]///home/user/video/movie.pvn" << std::endl;
    std::cout << "\tfile:///home/user/video/movie.avi" << std::endl;
    std::cout << "\tfiles:///home/user/seqiemce/foo%03d.jpeg" << std::endl;
    std::cout << "\tdc1394:[fmt=RGB24,size=640x480,fps=30,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tdc1394:[fmt=FORMAT7_1,size=640x480,pos=2+2,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tv4l:///dev/video0" << std::endl;
    std::cout << "\tconvert:[fmt=RGB24]//v4l:///dev/video0" << std::endl;
    std::cout << "\tmjpeg://http://127.0.0.1/?action=stream" << std::endl;
    std::cout << "\topenni:[img1=rgb]//" << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
