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
#include <thread>
#include <mutex>
#include <tdp/utils/threadedValue.hpp>
#include <tdp/camera/camera_poly.h>
#include <tdp/inertial/imu_obs.h>
#include <tdp/inertial/imu_outstream.h>
#include <tdp/drivers/inertial/3dmgx3_45.h>
#include <tdp/utils/Stopwatch.h>

typedef tdp::CameraPoly3<float> CameraT;
//typedef tdp::Camera<float> CameraT;

namespace tdp {

class PoseInterpolator {
 public:  
  PoseInterpolator()
  {}
  ~PoseInterpolator()
  {}

  /// Add a new <t,Pose> observation; 
  /// IMPORTANT: the assumption is that poses come in in chronological
  /// order.
  void Add(int64_t t, const SE3f& T) {
    std::lock_guard<std::mutex> lock(mutex_);
    ts_.push_back(t);
    Ts_.push_back(T);
  }

  SE3f operator[](int64_t t) {
    size_t i;
    // lock while we are looking for index then release
    mutex_.lock();
    for(i=0; i<ts_.size(); ++t) if (ts_[i]-t >=0) break;
    mutex_.unlock();

    if (i>0) {
      return Ts_[i-1].Exp(Ts_[i-1].Log(Ts_[i]) * (ts_[i-1]-t)*1e-6);
    } else {
      return Ts_[i];
    }
  }

 private:
  std::vector<int64_t> ts_;
  std::vector<SE3f> Ts_;
  std::mutex mutex_;
};


}

int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  std::string input_uri = std::string(argv[1]);
  std::string configPath = std::string(argv[2]);
  std::string output_uri = (argc > 3) ? std::string(argv[3]) : dflt_output_uri;

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

  tdp::Imu3DMGX3_45 imu("/dev/ttyACM0", 100);
  imu.Start();

  tdp::PoseInterpolator imuInterp;

  tdp::ThreadedValue<bool> receiveImu(true);
  std::thread receiverThread (
    [&]() {
      tdp::ImuObs imuObs;
      while(receiveImu.Get()) {
        if (imu.GrabNext(imuObs)) {
          imuInterp.Add(imuObs.t_host,
              tdp::SE3f(tdp::SO3f::R_rpy(imuObs.rpy).matrix(),
                Eigen::Vector3f(0,0,0)));
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });

  tdp::ImuOutStream imu_out("./testImu.pango");
  imu_out.Open(input_uri, imu.GetProperties());

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

  pangolin::View& viewImu = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewImu);
  // add a simple image viewer
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> logData("ui.log data",false,true);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    TICK("rgb collection");
    // get rgb image
    tdp::CollectRGB(rgbdStream2cam, gui, wSingle, hSingle, rgb,
        cudaMemcpyHostToHost);
//    for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
//      tdp::Image<tdp::Vector3bda> rgbStream;
//      if (!gui.ImageRGB(rgbStream, sId)) continue;
//      int32_t cId = rgbdStream2cam[sId]; 
//      tdp::Image<tdp::Vector3bda> rgb_i = rgb.GetRoi(0,cId*hSingle,
//          wSingle, hSingle);
//      rgb_i.CopyFrom(rgbStream,cudaMemcpyHostToHost);
//    }
    TOCK("rgb collection");
    TICK("depth collection");
    // get depth image
    tdp::CollectD<CameraT>(rgbdStream2cam, rig, gui, wSingle, hSingle, dMin, dMax,
        cuDraw, cuD);
//    for (size_t sId=0; sId < rgbdStream2cam.size(); sId++) {
//      tdp::Image<uint16_t> dStream;
//      if (!gui.ImageD(dStream, sId)) continue;
//      int32_t cId = rgbdStream2cam[sId]; 
//      tdp::Image<uint16_t> cuDraw_i = cuDraw.GetRoi(0,cId*hSingle,
//          wSingle, hSingle);
//      cuDraw_i.CopyFrom(dStream,cudaMemcpyHostToDevice);
//      // convert depth image from uint16_t to float [m]
//      tdp::Image<float> cuD_i = cuD.GetRoi(0, cId*hSingle, 
//          wSingle, hSingle);
//      //float depthSensorScale = depthSensor1Scale;
//      //if (cId==1) depthSensorScale = depthSensor2Scale;
//      //if (cId==2) depthSensorScale = depthSensor3Scale;
//      if (rig.depthScales_.size() > cId) {
//        float a = rig.scaleVsDepths_[cId](0);
//        float b = rig.scaleVsDepths_[cId](1);
//        // TODO: dont need to load this every time
//        cuScale.CopyFrom(rig.depthScales_[cId],cudaMemcpyHostToDevice);
//        tdp::ConvertDepthGpu(cuDraw_i, cuD_i, cuScale, a, b, dMin, dMax);
//      //} else {
//      //  tdp::ConvertDepthGpu(cuDraw_i, cuD_i, depthSensorScale, dMin, dMax);
//      }
//    }
    TOCK("depth collection");

    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
    // compute point cloud (on CPU)
    tdp::Depth2PC(d,cam,pc);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    viewImu.Activate(s_cam);
    tdp::SE3f T_wi; 
    pangolin::glDrawAxis<float>(T_wi.matrix(),0.8f);

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // ShowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // if we are recording
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    // finish this frame
    pangolin::FinishFrame();
  }
  receiveImu.Set(false);
  receiverThread.join();
  imu.Stop();
  imu_out.Close();
  return 0;
}

