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

#include <tdp/camera/rig.h>

#include <tdp/manifold/SE3.h>
#include <tdp/utils/threadedValue.hpp>
#include <tdp/camera/camera_poly.h>
#include <tdp/inertial/imu_obs.h>
#include <tdp/inertial/imu_outstream.h>
#include <tdp/drivers/inertial/3dmgx3_45.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/directional/hist.h>
#include <tdp/camera/ray.h>

#include <tdp/inertial/imu_factory.h>
#include <tdp/directional/spherical_coordinates.h>

#include <tdp/gl/gl_draw.h>
#include <tdp/gui/gui_base.hpp>

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

  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  rig.CorrespondOpenniStreams2Cams(streams);

  // optionally connect to IMU if it is found.
  tdp::ImuInterface* imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();
  tdp::ImuOutStream imu_out("./imu.pango");
  imu_out.Open(imu_input_uri, imu? imu->GetProperties() : pangolin::json::value());
  tdp::ImuInterpolator imuInterp(imu, &imu_out);
  imuInterp.Start();

  tdp::GuiBase gui(1200,800,video);

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

  pangolin::View& viewDirHist3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewDirHist3D);

  tdp::QuickView viewRGB(w,h);
  gui.container().AddDisplay(viewRGB);
  tdp::QuickView viewD(w,h);
  gui.container().AddDisplay(viewD);

  tdp::QuickView viewDirHist2D(400,400);
  gui.container().AddDisplay(viewDirHist2D);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> rays(w, h);

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

  pangolin::Var<bool> verbose("ui.verbose ",false,true);
  pangolin::Var<bool> collectStreams("ui.collect streams",true,true);

  pangolin::Var<float> histScale("ui.hist scale",1.,0.1,1.);
  pangolin::Var<bool> histLog("ui.hist log",false,true);
  pangolin::Var<bool> histShowEmpty("ui.show empty",true,true);
  pangolin::Var<bool> reset("ui.reset",true,false);

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
    Eigen::Matrix3f R_ir;
    R_ir << 0, 0,-1,
            0,-1, 0,
           -1, 0, 0;
    tdp::SE3f T_ir(R_ir,Eigen::Vector3f::Zero());
    tdp::SE3f T_wr = imuInterp.Ts_wi_[tNow]*T_ir; 
    TOCK("next frames");

    if (verbose) std::cout << "collecting rgb frames" << std::endl;
    if (collectStreams) {
      TICK("rgb collection");
      // get rgb image
      rig.CollectRGB(gui, rgb, cudaMemcpyHostToHost);
      TOCK("rgb collection");
      if (verbose) std::cout << "collecting depth frames" << std::endl;
      TICK("depth collection");
      // get depth image
      int64_t t_us;
      rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_us);
      d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
      TOCK("depth collection");
    }
    TICK("pc and normals");
    for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
      int32_t cId;
      cId = rig.dStream2cam_[sId]; 
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      tdp::Image<tdp::Vector3fda> cuPc_i = cuPc.GetRoi(0,
          rig.rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
      tdp::Image<float> cuD_i = cuD.GetRoi(0,
          rig.rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
      // compute point cloud from depth in rig coordinate system
      tdp::Depth2PCGpu(cuD_i, cam, T_rc, cuPc_i);
    }
    pc.CopyFrom(cuPc,cudaMemcpyDeviceToHost);
    TOCK("pc and normals");
    TICK("rays");
    for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
      int32_t cId;
      cId = rig.dStream2cam_[sId]; 
      CameraT cam = rig.cams_[cId];
      tdp::SE3f T_rc = rig.T_rcs_[cId];

      tdp::Image<tdp::Vector3fda> cuRays_i = cuRays.GetRoi(0,
          rig.rgbdStream2cam_[sId]*hSingle, wSingle, hSingle);
      // compute point cloud from depth in rig coordinate system
      tdp::ComputeCameraRays(cam, cuRays_i);
      tdp::TransformPc(T_wr*T_rc, cuRays_i);
    }
    dirHist.ComputeGpu(cuRays);
    TOCK("rays");

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
      pangolin::glDrawAxis<float>(T_wr.matrix(),0.8f);
      pangolin::glSetFrameOfReference(T_wr.matrix());
      pangolin::glDrawAxis(1.2f);
      pangolin::RenderVboCbo(vbo,cbo,true);
      pangolin::glUnsetFrameOfReference();
    }

    if (viewDirHist3D.IsShown()) {
      viewDirHist3D.Activate(s_cam);
      dirHist.geoGrid_.Render3D();
      dirHist.Render3D(histScale, histLog);
      rays.CopyFrom(cuRays,cudaMemcpyDeviceToHost);
      vbo.Upload(rays.ptr_,rays.SizeBytes(), 0);
      glColor4f(0,1,0,0.5f);
      pangolin::RenderVbo(vbo,true);
    }

    glDisable(GL_DEPTH_TEST);
    TOCK("draw 3D");
    // Draw 2D stuff
    // ShowFrames renders the raw input streams (in our case RGB and D)
    TICK("draw 2D");
    if (viewRGB.IsShown()) viewRGB.SetImage(rgb);
    if (viewD.IsShown()) viewD.SetImage(d);

    if (viewDirHist2D.IsShown()) {
      viewDirHist2D.ActivatePixelOrthographic();
      dirHist.Render2D(histScale, histLog, histShowEmpty);
      glColor4f(0,0,1,1);
      // plot some directions into the histogram to show where we are
      // currently collecting
      glPointSize(1);
      for (size_t sId=0; sId < rig.dStream2cam_.size(); sId++) {
        int32_t cId;
        cId = rig.dStream2cam_[sId]; 
        CameraT cam = rig.cams_[cId];
        tdp::SE3f T_wc = T_wr*rig.T_rcs_[cId];
        for (size_t u=0; u<wSingle; u += 20) {
          for (size_t v=0; v<hSingle; v += 20) {
            Eigen::Vector3f p1 = T_wc.rotation() * cam.Unproject(u,v,1.);
            Eigen::Vector3f phiTheta1 = tdp::ToSpherical(p1);
            int y1 = (-phiTheta1(0)+M_PI)*199/(2.*M_PI);
            int x1 = phiTheta1(1)*199/(M_PI);
            //pangolin::glDrawCircle(x1,y1,0.1);
            tdp::glDrawPoint(x1,y1);
          }
        }
      }
    }
    TOCK("draw 2D");

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // if we are recording
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
      imuInterp.StartRecording();
    } else {
      imuInterp.StopRecording();
    }
    // finish this frame
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }
  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;
  imu_out.Close();
  return 0;
}

