/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <thread>
#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/gldraw.h>

#include <Eigen/Dense>
#include <tdp/camera/rig.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/data/managed_volume.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/volume.h>
#include <tdp/data/circular_buffer.h>
#include <tdp/gl/gl_draw.h>
#include <tdp/gui/gui_base.hpp>
#include <tdp/gui/quickView.h>
#include <tdp/icp/icp.h>
#include <tdp/icp/icpRot.h>
#include <tdp/icp/icpGrad3d.h>
#include <tdp/icp/icpTexture.h>
#include <tdp/manifold/SE3.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/preproc/depth.h>
#include <tdp/preproc/normals.h>
#include <tdp/preproc/pc.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/manifold/SO3.h>
#include <tdp/preproc/grad.h>
#include <tdp/preproc/grey.h>
#include <tdp/preproc/mask.h>
#include <tdp/preproc/curvature.h>
#include <tdp/geometry/cosy.h>
#include <tdp/gl/shaders.h>
#include <tdp/utils/colorMap.h>
#include <tdp/camera/photometric.h>

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

int main( int argc, char* argv[] )
{
  std::string input_uri = "openni2://";
  std::string output_uri = "pango://video.pango";
  std::string calibPath = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
//    imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  }

  pangolin::Uri uri = pangolin::ParseUri(input_uri);
  if (!uri.scheme.compare("file")) {
    std::cout << uri.scheme << std::endl; 
    if (pangolin::FileExists(uri.url+std::string("imu.pango"))
     && pangolin::FileExists(uri.url+std::string("video.pango"))) {
//      imu_input_uri = input_uri + std::string("imu.pango");
      tsdfOutputPath = uri.url + tsdfOutputPath;
      input_uri = input_uri + std::string("video.pango");
    } else if (pangolin::FileExists(uri.url+std::string("video.pango"))) {
      input_uri = input_uri + std::string("video.pango");
    } 
  }

  std::cout << input_uri << std::endl;
  std::cout << imu_input_uri << std::endl;

  Stopwatch::getInstance().setCustomSignature(82043984912);

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  // optionally connect to IMU if it is found.
  tdp::ImuInterface* imu = nullptr; 
  if (imu_input_uri.size() > 0) 
    imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();
  tdp::ImuInterpolator imuInterp(imu,nullptr);
  imuInterp.Start();

  tdp::GuiBase gui(1200,800,video);
  gui.container().SetLayout(pangolin::LayoutEqual);

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,false);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
  } else {
    return 2;
  }

  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = (w+w%64); // for convolution
  size_t hc = rig.NumCams()*(h+h%64);
  wc += wc%64;
  hc += hc%64;

  tdp::Camera<float> camView(Eigen::Vector4f(220,220,319.5,239.5)); 
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);

  pangolin::View& viewAssoc = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewAssoc);

  pangolin::View& containerTracking = pangolin::Display("tracking");
  containerTracking.SetLayout(pangolin::LayoutEqual);
  tdp::QuickView viewModel(wc, hc);
  containerTracking.AddDisplay(viewModel);
  tdp::QuickView viewCurrent(wc, hc);
  containerTracking.AddDisplay(viewCurrent);
  tdp::QuickView viewMask(wc, hc);
  containerTracking.AddDisplay(viewMask);
  gui.container().AddDisplay(containerTracking);

  containerTracking.Show(false);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> rgb_m(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);

  tdp::ManagedHostImage<tdp::Vector3fda> pcFull_m(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wc,hc);
  tdp::ManagedDeviceImage<float> cuGrey(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedDevicePyramid<uint8_t,3> cuPyrMask(wc, hc);
  tdp::ManagedDeviceImage<uint8_t> cuMask(wc, hc);
  tdp::ManagedHostImage<uint8_t> mask(wc, hc);

  // ICP stuff
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

  tdp::ManagedHostImage<tdp::Vector3fda> pc_c;
  tdp::ManagedHostImage<tdp::Vector3fda> n_c;

  tdp::ManagedHostImage<tdp::Vector3fda> pc_m;
  tdp::ManagedHostImage<tdp::Vector3fda> n_m;

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",6.,0.1,10.);

  pangolin::Var<float> subsample("ui.subsample %",0.002,0.0001,.01);

  pangolin::Var<float> scale("ui.scale %",0.1,0.1,1);

  pangolin::Var<float> angleThr("ui.angle Thr",15, 0, 90);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.01,0,0.3);
  pangolin::Var<float> distThr("ui.dist Thr",0.1,0,0.3);
  pangolin::Var<int> maxIt("ui.max iter",20, 1, 20);

  pangolin::Var<int>   W("ui.W ",4,1,15);
  pangolin::Var<int>   dispLvl("ui.disp lvl",0,0,2);

  pangolin::Var<bool> showPlanes("ui.show planes",false,true);
  pangolin::Var<bool> showPcModel("ui.show model",false,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",false,true);
  pangolin::Var<bool> showFullPc("ui.show full",false,true);

  tdp::SE3f T_wc_0;
  tdp::SE3f T_wc = T_wc_0;
  std::vector<tdp::SE3f> T_wcs;
  tdp::SE3f T_mc; // current to model

  gui.verbose = true;
  if (gui.verbose) std::cout << "starting main loop" << std::endl;

  pangolin::GlBuffer vbo_w(pangolin::GlArrayBuffer,1000000,GL_FLOAT,3);
  tdp::ManagedHostCircularBuffer<tdp::Vector3fda> pc_w(1000000);
  pc_w.Fill(tdp::Vector3fda(NAN,NAN,NAN));

  size_t frame = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {

    if (!gui.paused() && frame > 0) {
      pcs_m.CopyFrom(pcs_c);
      pcFull_m.CopyFrom(pcs_m.GetImage(0));
      pc_m.ResizeCopyFrom(pc_c); 
      n_m.ResizeCopyFrom(n_c); 
      rgb_m.CopyFrom(rgb);

      for (size_t i=0; i<pc_c.Area(); ++i) pc_c[i] = T_wc * pc_c[i];
      pc_w.Insert(pc_c);
      vbo_w.Upload(pc_w.ptr_, pc_w.SizeBytes(), 0);
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();

    int64_t t_host_us_d = 0;
    if (gui.verbose) std::cout << "setup pyramids" << std::endl;
    TICK("Setup Pyramids");
    if (gui.verbose) std::cout << "collect d" << std::endl;
    rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
    if (gui.verbose) std::cout << "compute pc" << std::endl;
    rig.ComputePc(cuD, true, pcs_c);
    if (gui.verbose) std::cout << "compute n" << std::endl;
    rig.ComputeNormals(cuD, true, ns_c);
    if (gui.verbose) std::cout << "collect rgb" << std::endl;
    rig.CollectRGB(gui, rgb) ;

    if (!gui.paused() || subsample.GuiChanged()) {
      tdp::RandomMaskCpu(mask, subsample);
      cuMask.CopyFrom(mask);
      tdp::ConstructPyramidFromImage(cuMask, cuPyrMask);
    }
    TOCK("Setup Pyramids");

    pc.CopyFrom(pcs_c.GetImage(0));
    size_t numObs = 0;
    for (size_t i=0; i<mask.Area(); ++i) {
      if (mask[i]) numObs++;
    }
    std::cout << numObs << std::endl;
    pc_c.Reinitialise(numObs);
    n_c.Reinitialise(numObs);
    size_t j=0;
    for (size_t i=0; i<mask.Area(); ++i) {
      if (mask[i]) {
        uint32_t u0 = i%wc;
        uint32_t v0 = i/wc;
        pc_c[j] = pc(u0,v0);
        if (!tdp::NormalViaScatter(pc, u0, v0, W, n_c[j++])) {
          n_c[j-1] << NAN,NAN,NAN;
        }
      }
    }

    std::vector<std::pair<size_t, size_t>> assoc;
    T_mc = tdp::SE3f();
    for (size_t it = 0; it < maxIt; ++it) {
      Eigen::Matrix<float,6,6> A = Eigen::Matrix<float,6,6>::Zero();
      Eigen::Matrix<float,6,1> b = Eigen::Matrix<float,6,1>::Zero();
      Eigen::Matrix<float,6,1> Ai = Eigen::Matrix<float,6,1>::Zero();
      float bi = 0.;
      float err = 0.;
      uint32_t numInl = 0;
      float dotThr = cos(angleThr*M_PI/180.);
      for (size_t i=0; i<pc_m.Area(); ++i) {
        for (size_t j=0; j<pc_c.Area(); ++j) {
          if (tdp::IsValidNormal(n_m[i]) && tdp::IsValidNormal(n_c[j]) 
              && fabs(n_m[i].dot(n_c[j])) > dotThr) {
            float p2pl = n_m[i].dot(pc_m[i] - T_mc*pc_c[j]);
            float dist = (pc_m[i] - T_mc*pc_c[j]).norm();
            if (tdp::IsValidData(pc_m[i]) && tdp::IsValidData(pc_c[j]) 
                && fabs(p2pl) < p2plThr
                && dist < distThr*pc_m[i](2)) {
//              std::cout << acos(n_m[i].dot(n_c[j]))*180./M_PI 
//                << " " << p2pl << std::endl;
              // match
              Eigen::Vector3f n_m_in_c = T_mc.rotation().Inverse()*n_m[i];
              Ai.topRows<3>() = pc_c[j].cross(n_m_in_c); 
              Ai.bottomRows<3>() = n_m_in_c; 
              bi = p2pl;
              A += Ai * Ai.transpose();
              b += Ai * bi;
              numInl ++;
              err += p2pl;
              assoc.emplace_back(i,j);
            }
          }
        }
      }
      Eigen::Matrix<float,6,1> x = Eigen::Matrix<float,6,1>::Zero();
      if (numInl > 10) {
        // solve for x using ldlt
        x = (A.cast<double>().ldlt().solve(b.cast<double>())).cast<float>(); 
        T_mc = T_mc * tdp::SE3f::Exp_(x);
      }
      if (gui.verbose) {
        std::cout << " it " << it 
          << ": err=" << err
          << "\t# inliers: " << numInl
          << "\t|x|: " << x.topRows(3).norm()*180./M_PI 
          << " " <<  x.bottomRows(3).norm()
          << std::endl;
      }
    }

    if (!gui.paused()) {
      T_wc = T_wc*T_mc;
      T_wcs.push_back(T_wc);
    }

    frame ++;

    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    TICK("Draw 3D");
    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);

      pangolin::glDrawAxis(T_wc.matrix(), 0.05f);
      glColor4f(1.,1.,0.,0.6);
      glDrawPoses(T_wcs,20);

      if (showFullPc) {
        glColor4f(0.,1.,1.,0.6);
        pangolin::RenderVbo(vbo_w);
      }

      pangolin::glSetFrameOfReference((T_wc*T_mc.Inverse()).matrix());
      vbo.Reinitialise(pangolin::GlArrayBuffer, pc_m.Area(), GL_FLOAT,
          3, GL_DYNAMIC_DRAW);
      vbo.Upload(pc_m.ptr_, pc_m.SizeBytes(), 0);
      glColor3f(1,0,0);
      pangolin::RenderVbo(vbo);
      pangolin::glUnsetFrameOfReference();

      // draw sampled pc points and mean curvature
//      vbo.Reinitialise(pangolin::GlArrayBuffer, pc_c.Area(), GL_FLOAT, 3,
//          GL_DYNAMIC_DRAW);
//      vbo.Upload(pc_c.ptr_, pc_c.SizeBytes(), 0);
//      glColor3f(1,0,0);
      pangolin::glSetFrameOfReference(T_wc.matrix());
//      pangolin::RenderVbo(vbo);
      glColor3f(1,1,0);
      for (size_t i=0; i<n_c.Area(); ++i) {
        tdp::glDrawLine(pc_c[i], pc_c[i] + scale*n_c[i]);
      }
      pangolin::glUnsetFrameOfReference();
      if (showPlanes) {
        for (size_t i=0; i<n_c.Area(); ++i) {
          Eigen::Matrix3f R = tdp::OrthonormalizeFromYZ(
              Eigen::Vector3f(0,1,0), n_c[i].normalized());
          tdp::SE3f T(R, pc_c[i]); 
          pangolin::glSetFrameOfReference((T_wc*T).matrix());
          pangolin::glDrawAxis(0.05f);
          pangolin::glDraw_z0(0.01,10);
          pangolin::glUnsetFrameOfReference();
        }
      }

      // render current camera second in the propper frame of
      // reference
      if (showPcCurrent) {
        vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(), GL_FLOAT,
            3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
        pangolin::glSetFrameOfReference(T_wc.matrix());
        if(dispLvl == 0){
          cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
          pangolin::RenderVboCbo(vbo, cbo, true);
        } else {
          glColor3f(1,0,0);
          pangolin::RenderVbo(vbo);
        }
        pangolin::glUnsetFrameOfReference();
      }
    }

    if (viewAssoc.IsShown()) {
      viewAssoc.Activate(s_cam);

      pangolin::glSetFrameOfReference((T_mc).matrix());
      pangolin::glDrawAxis(0.1f);
      vbo.Reinitialise(pangolin::GlArrayBuffer, pc_c.Area(), GL_FLOAT,
          3, GL_DYNAMIC_DRAW);
      vbo.Upload(pc_c.ptr_, pc_c.SizeBytes(), 0);
      glColor3f(0,1,0);
      pangolin::RenderVbo(vbo);

      if (showPcCurrent) {
        vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(), GL_FLOAT,
            3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
        cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
        pangolin::RenderVboCbo(vbo, cbo, true);
      }

      pangolin::glUnsetFrameOfReference();

      pangolin::glDrawAxis(0.3f);
      vbo.Reinitialise(pangolin::GlArrayBuffer, pc_m.Area(), GL_FLOAT,
          3, GL_DYNAMIC_DRAW);
      vbo.Upload(pc_m.ptr_, pc_m.SizeBytes(), 0);
      glColor3f(1,0,0);
      pangolin::RenderVbo(vbo);

      if (showPcModel) {
        vbo.Reinitialise(pangolin::GlArrayBuffer, pcFull_m.Area(), GL_FLOAT,
            3, GL_DYNAMIC_DRAW);
        vbo.Upload(pcFull_m.ptr_, pcFull_m.SizeBytes(), 0);
        cbo.Upload(rgb_m.ptr_, rgb_m.SizeBytes(), 0);
        pangolin::RenderVboCbo(vbo, cbo, true);
      }

      glColor4f(0,1,1,0.3);
      for (const auto& ass : assoc) {
        tdp::glDrawLine(pc_m[ass.first], pc_c[ass.second]);
      }

    }

    TOCK("Draw 3D");
    if (gui.verbose) std::cout << "draw 2D" << std::endl;
    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (containerTracking.IsShown()) {
      if (viewModel.IsShown()) {
        viewModel.SetImage(rgb_m);
      }
      if (viewCurrent.IsShown()) {
        viewCurrent.SetImage(rgb);
      }
      if (viewMask.IsShown()) {
        viewMask.SetImage(mask);
      }
    }

    TOCK("Draw 2D");

    if (gui.verbose) std::cout << "finished one iteration" << std::endl;
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }

  imuInterp.Stop();
  if (imu) imu->Stop();
  delete imu;
  std::this_thread::sleep_for(std::chrono::microseconds(500));
  return 0;
}

