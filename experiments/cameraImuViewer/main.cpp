/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
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
#include <pangolin/utils/timer.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/gui/gui.hpp>
#include <tdp/camera/rig.h>

#include <tdp/manifold/SE3.h>
#include <tdp/utils/threadedValue.hpp>
#include <tdp/camera/camera_poly.h>
#include <tdp/inertial/imu_obs.h>
#include <tdp/inertial/imu_outstream.h>
#include <tdp/drivers/inertial/3dmgx3_45.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/pose_interpolator.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/directional/hist.h>
#include <tdp/camera/ray.h>

typedef tdp::CameraPoly3<float> CameraT;
//typedef tdp::Camera<float> CameraT;


int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  std::string input_uri = std::string(argv[1]);
  std::string configPath = std::string(argv[2]);
  std::string imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  std::string output_uri = (argc > 4) ? std::string(argv[4]) : dflt_output_uri;

  // Read rig file
  tdp::Rig<CameraT> rig;
  if (!rig.FromFile(configPath, false)) return 1;

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 2;
  }

  std::vector<int32_t> rgbStream2cam;
  std::vector<int32_t> dStream2cam;
  std::vector<int32_t> rgbdStream2cam;
  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  tdp::CorrespondOpenniStreams2Cams(streams,rig,rgbStream2cam,
      dStream2cam, rgbdStream2cam);

  // optionally connect to IMU if it is found.
  tdp::ImuInterface* imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();

  tdp::PoseInterpolator imuInterp;

  tdp::ImuOutStream imu_out("./imu.pango");
  imu_out.Open(imu_input_uri, imu? imu->GetProperties() : pangolin::json::value());

  tdp::GUI gui(1200,800,video);

  size_t wSingle = video.Streams()[0].Width();
  size_t hSingle = video.Streams()[0].Height();
  wSingle += wSingle%64;
  hSingle += hSingle%64;
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

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPc(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuRays(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> logData("ui.log data",false,true);
  pangolin::Var<bool> verbose("ui.verbose ",false,true);
  pangolin::Var<bool> collectStreams("ui.collect streams",true,true);

  pangolin::Var<float> histScale("ui.hist scale",1.,0.1,1.);
  pangolin::Var<bool> reset("ui.reset",true,false);

  tdp::ThreadedValue<bool> receiveImu(true);
  tdp::ThreadedValue<size_t> numReceived(0);
  std::thread receiverThread (
    [&]() {
      tdp::ImuObs imuObs;
      tdp::ImuObs imuObsPrev;
      while(receiveImu.Get()) {
        if (imu && imu->GrabNext(imuObs)) {

          Eigen::Matrix<float,6,1> se3 = Eigen::Matrix<float,6,1>::Zero();
          se3.topRows(3) = imuObs.omega;
          if (numReceived.Get() == 0) {
            imuInterp.Add(imuObs.t_host, tdp::SE3f());
          } else {
            int64_t dt_ns = imuObs.t_device - imuObsPrev.t_device;
            imuInterp.Add(imuObs.t_host, se3, dt_ns);
          }
          imuObsPrev = imuObs;
          numReceived.Increment();

          if (imu && video.IsRecording()) {
            imu_out.WriteStream(imuObs);
          }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });

  tdp::GeodesicHist<3> dirHist;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (pangolin::Pushed(reset)) {
      dirHist.Reset();
    }

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    TICK("next frames");
    gui.NextFrames();
    int64_t tNow = pangolin::Time_us(pangolin::TimeNow())*1000;
    TOCK("next frames");

    if (verbose) std::cout << "collecting rgb frames" << std::endl;
    if (collectStreams) {
      TICK("rgb collection");
      // get rgb image
      tdp::CollectRGB(rgbdStream2cam, gui, wSingle, hSingle, rgb,
          cudaMemcpyHostToHost);
      TOCK("rgb collection");
      if (verbose) std::cout << "collecting depth frames" << std::endl;
      TICK("depth collection");
      // get depth image
      tdp::CollectD<CameraT>(rgbdStream2cam, rig, gui, wSingle,
          hSingle, dMin, dMax, cuDraw, cuD);
      d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
      TOCK("depth collection");
    }
    TICK("pc and normals");
    for (size_t sId=0; sId < dStream2cam.size(); sId++) {
      int32_t cId;
      cId = dStream2cam[sId]; 
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      tdp::Image<tdp::Vector3fda> cuPc_i = cuPc.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);
      tdp::Image<float> cuD_i = cuD.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);
      // compute point cloud from depth in rig coordinate system
      tdp::Depth2PCGpu(cuD_i, cam, T_rc, cuPc_i);
    }
    pc.CopyFrom(cuPc,cudaMemcpyDeviceToHost);
    TOCK("pc and normals");
    TICK("rays");
    for (size_t sId=0; sId < dStream2cam.size(); sId++) {
      int32_t cId;
      cId = dStream2cam[sId]; 
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      tdp::Image<tdp::Vector3fda> cuRays_i = cuRays.GetRoi(0,
          rgbdStream2cam[sId]*hSingle, wSingle, hSingle);
      // compute point cloud from depth in rig coordinate system
      tdp::ComputeCameraRays(cam, cuRays_i);
      tdp::TransformPc(T_rc, cuRays_i);
    }
    dirHist.ComputeGpu(cuRays);
    TOCK("rays");

    Eigen::Matrix3f R_ir;
    R_ir << 0, 0,-1,
            0,-1, 0,
           -1, 0, 0;
    tdp::SE3f T_ir(R_ir,Eigen::Vector3f::Zero());
    tdp::SE3f T_wi = imuInterp[tNow]*T_ir; 
    // Draw 3D stuff
    TICK("draw 3D");
    glEnable(GL_DEPTH_TEST);

    if (d_cam.IsShown()) {
      d_cam.Activate(s_cam);
      // draw the axis
      vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
      // render point cloud
      pangolin::glDrawAxis(1.0f);
      pangolin::glDrawAxis<float>(T_wi.matrix(),0.8f);
      pangolin::glSetFrameOfReference(T_wi.matrix());
      pangolin::glDrawAxis(1.2f);
      pangolin::RenderVboCbo(vbo,cbo,true);
      pangolin::glUnsetFrameOfReference();
    }

    if (viewDirHist3D.IsShown()) {
      dirHist.Render3D(histScale, false);
    }

    glDisable(GL_DEPTH_TEST);
    TOCK("draw 3D");
    // Draw 2D stuff
    // ShowFrames renders the raw input streams (in our case RGB and D)
    TICK("draw 2D");
    gui.ShowFrames();
    TOCK("draw 2D");

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
  receiveImu.Set(false);
  receiverThread.join();
  if (imu) imu->Stop();
  imu_out.Close();
  return 0;
}

