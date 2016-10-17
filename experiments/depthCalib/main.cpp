/* This code is based on Calibu/applications/calib/main.cpp
 * Since that file itself does not contain a license header I just
 * state that parts of the code for computing the pose relative to the
 * target as well as vor displaying the target are from Calibu which is
 * under Apache-2.0 license. 
 *
 * Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <fstream>
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
#include <tdp/camera/ray.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <calibu/calib/Calibrator.h>
#include <calibu/cam/camera_models_kb4.h>
#include <calibu/image/ImageProcessing.h>
#include <calibu/target/TargetGridDot.h>
#include <calibu/target/RandomGrid.h>
#include <calibu/gl/Drawing.h>
#include <calibu/pose/Pnp.h>
#include <calibu/conics/ConicFinder.h>

#include <tdp/gui/gui.hpp>
#include <tdp/preproc/grey.h>
#include <tdp/calibration/PnP.h>
#include <tdp/camera/rig.h>
#include <tdp/utils/Stopwatch.h>

int main( int argc, char* argv[] )
{
  std::string input_uri = "openni2://";
  std::string output_uri = "pango://video.pango";
  std::string calibPath = "";

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
  }

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  tdp::GUI gui(1200,800,video);
  Stopwatch::getInstance().setCustomSignature(1237249810);

  size_t w = video.Streams()[gui.iD[0]].Width();
  size_t h = video.Streams()[gui.iD[0]].Height();

  tdp::QuickView viewGrey(w,h);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewMask(w,h);
  gui.container().AddDisplay(viewMask);
  tdp::QuickView viewScale(w,h);
  gui.container().AddDisplay(viewScale);
  tdp::QuickView viewScaleN(w,h);
  gui.container().AddDisplay(viewScaleN);

  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
  pangolin::DataLog logScale;
  pangolin::Plotter plotScale(&logScale, -100.f,1.f, 0.7f, 1.3f, 
      1.f, 0.1f);
  plotters.AddDisplay(plotScale);
  pangolin::DataLog logDepth;
  pangolin::Plotter plotDepth(&logDepth, -100.f,1.f, 0.1f, 4.f, 1.f, 0.1f);
  plotters.AddDisplay(plotDepth);
  gui.container().AddDisplay(plotters);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);

  tdp::Rig<tdp::CameraPoly3<float>> rig;
  rig.FromFile(calibPath,true);

  std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
  rig.CorrespondOpenniStreams2Cams(streams);

  // camera model for computing point cloud and normals
  tdp::CameraPoly3<float> camRGB = rig.cams_[rig.rgbStream2cam_[0]];
  tdp::CameraPoly3<float> camD = camRGB; //rig.cams_[dStream2cam[0]];
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<uint8_t> grey(w,h);
  tdp::ManagedHostImage<uint8_t> mask(w,h);
  tdp::ManagedHostImage<float> scaleN(w,h);
  tdp::ManagedHostImage<float> scale(w,h);
  tdp::ManagedDeviceImage<float> cuScale(w,h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> estimateScale("ui.est scale",true,true);
  pangolin::Var<bool> applyScale("ui.apply scale",true,true);
  pangolin::Var<bool> resetScale("ui.reset scale est",false,false);
  pangolin::Var<int> patchBoundary("ui.patch boundary",3,0,10);
  pangolin::Var<float> numScaleObs("ui.# obs",1000.f,100.f,2000.f);
  pangolin::Var<bool> saveScaleCalib("ui.save scale est",false,false);
  pangolin::Var<bool> logScaleVsDist("ui.log scale vs dist",false,true);

  int grid_rows = 12;
  int grid_cols = 24;
  uint32_t grid_seed = 57;
  double grid_spacing = 0.077;

  bool lymphedema = true;
  if (lymphedema) {
    std::cout << "running with lymphedema parameters!" << std::endl;
    grid_rows = 12;
    grid_cols = 16;
    grid_seed = 76;
    grid_spacing = 0.0165;
    depthSensorScale = 1e-4;
  }

  const Eigen::Vector2i grid_size(grid_rows, grid_cols);
  calibu::TargetGridDot target( grid_spacing, grid_size, grid_seed );

  calibu::ConicFinder conic_finder;
  conic_finder.Params().conic_min_area = 4.0;
  conic_finder.Params().conic_min_density = 0.6;
  conic_finder.Params().conic_min_aspect = 0.2;

  calibu::ImageProcessing image_processing(w, h);
  image_processing.Params().black_on_white = true;
  image_processing.Params().at_threshold = 0.9;
  image_processing.Params().at_window_ratio = 30.0;

  std::vector<tdp::SE3f> T_hws;
  
  if (rig.cuDepthScales_.size() > rig.rgbdStream2cam_[0]) {
    scale.CopyFrom(rig.cuDepthScales_[rig.rgbdStream2cam_[0]], cudaMemcpyDeviceToHost);
    scaleN.Fill(100.f);
  } else {
    resetScale = true;
  }

  std::ofstream log;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (pangolin::Pushed(resetScale)) {
      scale.Fill(depthSensorScale);
      scaleN.Fill(0.);
    }
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    TICK("Get data");
    // get rgb image
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    tdp::Rgb2GreyCpu<uint8_t>(rgb, grey, 1.0f);
    // get depth image
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    // copy raw image to gpu
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    TOCK("Get data");
    TICK("scale depth");
    // convet depth image from uint16_t to float [m]
    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
    TOCK("scale depth");

    TICK("conics");
    image_processing.Process(grey.ptr_, grey.w_, grey.h_, grey.pitch_);
    conic_finder.Find( image_processing );
    const std::vector<calibu::Conic,
          Eigen::aligned_allocator<calibu::Conic> >& conics =
            conic_finder.Conics();
    std::vector<int> ellipse_target_map;

    bool tracking_good = target.FindTarget(image_processing,
        conic_finder.Conics(), ellipse_target_map);
    TOCK("conics");

    tdp::SE3f T_hw;
    if(tracking_good) {
      // Generate map and point structures
      std::vector<Eigen::Vector2d,
        Eigen::aligned_allocator<Eigen::Vector2d> > ellipses;
      for( size_t i=0; i < conics.size(); ++i ) {
        ellipses.push_back(conics[i].center);
      }

      // find camera pose given intrinsics
      std::vector<int> inliers = tdp::PnPRansac(
          camRGB, ellipses, target.Circles3D(),
          ellipse_target_map,
          0, 0, T_hw);
      T_hws.push_back(T_hw);

      TICK("generating mask");
      mask.Fill(0);
      for( size_t i=0; i < conics.size(); ++i ) {
        if (ellipse_target_map[i] > 0) {
          for (size_t y=std::max(0,conics[i].bbox.y1-patchBoundary); 
              y<=std::min((int)h-1,conics[i].bbox.y2+patchBoundary); ++y) {
            for (size_t x=std::max(0,conics[i].bbox.x1-patchBoundary); 
              x<=std::min((int)w-1,conics[i].bbox.x2+patchBoundary); ++x) {
              mask(x,y) = 255;
            }
          }
        }
      }
      TOCK("generating mask");

      TICK("computing scale");
      tdp::SE3f T_wh = T_hw.Inverse();
      float avgScale = 0.f;
      float avgDepth = 0.f;
      float numScale = 0.f;
      for (size_t i=0; i<mask.Area(); ++i) {
        if (mask[i] > 0 && !std::isnan(d[i]) && dMin <= d[i] && d[i] <= dMax) {
          tdp::Rayfda ray_d(tdp::Vector3fda::Zero(),
              camD.Unproject(i%w, i/w, 1.f));
          // ray of depth image d in world coordinates
          tdp::Rayfda ray_w = ray_d.Transform(T_wh);
          // float d_true = - ray_w.dot(n)/(ray_w.dir.dot(n));
          // above is equivalent to belwo because n = (0,0,-1)
          float d_true = -ray_w.p(2)/ray_w.dir(2);
          // true depth over observed depth
          float scale_i = depthSensorScale*d_true/d[i];
          if (estimateScale) {
            // dot product between plane normal and ray direction
            float w_i = ray_w.dir(2)/ray_w.dir.norm();
            scale[i] = (scale[i]*scaleN[i]+scale_i*w_i)/(scaleN[i]+w_i);
            scaleN[i] = std::min(scaleN[i]+w_i,numScaleObs.Get());
          }
          //avgDepth += scale[i]*d[i]/depthSensorScale;
          avgDepth += scale[i]*d[i]/depthSensorScale;
          avgScale += scale_i/scale[i];
          numScale ++;
        }
      }
      avgScale /= numScale;
      avgDepth /= numScale;
      TOCK("computing scale");

      logDepth.Log(avgDepth);
      logScale.Log(avgScale);
      plotDepth.ScrollView(1,0);
      plotScale.ScrollView(1,0);
      std::cout << " avg scale: " << avgScale << "\tavg depth: " << avgDepth 
        << std::endl;
      if (logScaleVsDist.GuiChanged() && logScaleVsDist) {
        std::stringstream ss;
        ss << "scaleVsDist_" << rig.serials_[rig.rgbdStream2cam_[0]] << ".csv";
        log.open(ss.str());
      }
      if (logScaleVsDist) {
        log << avgScale << " " << avgDepth << std::endl;
      }
      if (logScaleVsDist.GuiChanged() && !logScaleVsDist) {
        log.close();
      }
    }

    if (pangolin::Pushed(saveScaleCalib)) {
      std::string path = CONFIG_DIR+std::string("depthCalib.png");
      if (rig.depthScalePaths_.size() > rig.rgbdStream2cam_[0]) {
        path = rig.depthScalePaths_[rig.rgbdStream2cam_[0]];
      }
      pangolin::Image<uint8_t> scale8bit(w*sizeof(float),h,
          w*sizeof(float),(uint8_t*)scale.ptr_);
      pangolin::SaveImage(scale8bit,pangolin::VideoFormatFromString("GRAY8"),
          path);
      std::cout << "saved depth scale calib to " << path << std::endl;
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    if (d_cam.IsShown()) {
      d_cam.Activate(s_cam);
      // draw the axis
      pangolin::glDrawAxis(1.0);

      if (applyScale) {
        float a = rig.scaleVsDepths_[rig.rgbdStream2cam_[0]](0);
        float b = rig.scaleVsDepths_[rig.rgbdStream2cam_[0]](1);
//        std::cout << a << " " << b<< std::endl;
        cuScale.CopyFrom(scale,cudaMemcpyHostToDevice);
        tdp::ConvertDepthGpu(cuDraw, cuD, cuScale, a, b, dMin, dMax);
        d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
      }
      tdp::Depth2PC(d,camD,pc);

      vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
      // render point cloud
      if(tracking_good) {
        pangolin::glSetFrameOfReference(T_hws.back().Inverse().matrix());
        pangolin::RenderVboCbo(vbo,cbo,true);
        pangolin::glUnsetFrameOfReference();
        pangolin::glDrawFrustrum(camD.GetKinv(),w,h,T_hw.Inverse().matrix(),0.05f);
      }
      calibu::glDrawTarget(target, Eigen::Vector2d(0,0), 1.0, 0.8, 1.0);
      for (size_t i=0; i<T_hws.size(); ++i) {
        pangolin::glDrawAxis(T_hws[i].Inverse().matrix(),100.f*depthSensorScale);
      }
    }

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();
    viewGrey.SetImage(grey);
    viewMask.SetImage(mask);
    viewScale.SetImage(scale);
    viewScaleN.SetImage(scaleN);

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

  return 0;
}
