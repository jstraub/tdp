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
#include <tdp/camera/camera_poly.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/rig.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/normals.h>
#include <tdp/data/circular_buffer.h>
#include <tdp/preproc/plane.h>
#include <tdp/gl/gl_draw.h>

#include <tdp/gui/gui.hpp>
#include <tdp/io/tinyply.h>

typedef tdp::CameraPoly3f CameraT;

int main( int argc, char* argv[] )
{
  std::string input_uri = std::string(argv[1]);
  std::string calibPath = "";
  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
//    imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  }

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, "./video.pango");
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,true);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
  } else {
    return 2;
  }
  CameraT cam = rig.cams_[1]; // for NYU

  tdp::GUI gui(1200,800,video);

  size_t w = video.Streams()[gui.iD[0]].Width();
  size_t h = video.Streams()[gui.iD[0]].Height();
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
  // add a simple image viewer
  tdp::QuickView viewSeg(w,h);
  tdp::QuickView viewCurv(w,h);
  gui.container().AddDisplay(viewSeg);
  gui.container().AddDisplay(viewCurv);
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector4fda> dpc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> n(w, h);
  tdp::ManagedHostImage<float> curv(w, h);
  tdp::ManagedHostImage<uint16_t> z(w, h);
  tdp::ManagedHostImage<uint8_t> mask(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);
  pangolin::Var<float> scale("ui.scale",0.1,0.1,0.2);
  pangolin::Var<float> p2plThr("ui.p2pl Thr",0.06,0.1,0.3);
  pangolin::Var<float> angThr("ui.ang Thr",15.,10.,30.);
  pangolin::Var<float> distThr("ui.dist Thr",10.,0.5,3.);
  pangolin::Var<float> curvThr("ui.curv Thr",0.06,0.01,0.1);
  pangolin::Var<float> inlierThr("ui.plane inl Thr",0.5, 0.5, 1.0);
  pangolin::Var<int> W("ui.W",9,1,15);
  pangolin::Var<int> nPlanes("ui.nPlanes",1000,100,1000);

  pangolin::Var<bool> useVoting("ui.use voting",false,true);

  tdp::ManagedHostCircularBuffer<tdp::Plane> pls(100000);

  std::stringstream ss;
  if (useVoting) {
    ss << "./sparsePlaneSeg_voting_" << nPlanes << ".csv";
  } else {
    ss << "./sparsePlaneSeg_LS_" << nPlanes << ".csv";
  }
  std::ofstream out(ss.str());

  // Stream and display video
  while(!pangolin::ShouldQuit() && !gui.finished())
  {
    float dotThr = cos(angThr*M_PI/180.);

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    // get rgb image
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    // get depth image
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    // copy raw image to gpu
    cuDraw.CopyFrom(dRaw);
    // convet depth image from uint16_t to float [m]
    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    d.CopyFrom(cuD);
    // compute point cloud (on CPU)
    tdp::Depth2PC(d,cam,pc);

    if (useVoting) {
      tdp::NormalsViaVoting(pc, W, 1, inlierThr, dpc, n, curv);
    } else {
      tdp::NormalsViaScatterUnconstrained(pc, W, 1, n);
    }
    std::cout << "normals computed" << std::endl;

    std::vector<int> ids(w*h);
    std::iota(ids.begin(), ids.end(), 0);
    std::random_shuffle(ids.begin(), ids.end());
    pls.MarkRead(); 
    z.Fill(0);
    mask.Fill(0);
    uint32_t numCovered = 0;
    uint32_t j = 0;
    for (int m=0; m<nPlanes; ++m) {
      tdp::Plane pl; 
      pl.curvature_ = 1.;
      while(pl.curvature_ > curvThr && j < ids.size()) {
        int l = ids[j++];
        if (tdp::IsValidData(pc[l]) && tdp::IsValidData(n[l])) {
          pl.n_ = n[l];
          pl.p_ = pc[l];
          pl.curvature_ = curv[l];
        }
        for (size_t u=std::max(0,(int)(l%w)-W); u < std::min((int)w, (int)(l%w)+W); ++u)
          for (size_t v=std::max(0,(int)(l/w)-W); v < std::min((int)h, (int)(l/w)+W); ++v)
            if (tdp::IsValidData(pc(u,v)) && tdp::IsValidData(n(u,v)))
              mask(u,v) = 1;
//        tdp::NormalViaVoting(pc, l%w, l/w, W, inlierThr, dpc, pl.n_,
//            pl.curvature_, pl.p_);
      }
      pls.Insert(pl);
      for (size_t i=0; i<z.Area(); ++i) {
        if (z[i] == 0) {
          tdp::Plane plO;
          plO.p_ = pc[i];
          plO.n_ = n[i];
          if (pl.Close(plO, dotThr, distThr, p2plThr)) {
            z[i] = m+1;
            numCovered ++;
          }
        }
      }
      uint32_t numTouched = 0;
      for (size_t i=0; i<mask.Area(); ++i) if(mask[i]>0) numTouched++;
      out << numCovered << " " << numTouched << " ";
    }
    uint32_t N = 0;
//    for (size_t i=0; i<pc.Area(); ++i) if (tdp::IsValidData(pc[i]) N ++;
    for (size_t i=0; i<n.Area(); ++i) if (tdp::IsValidData(n[i])) N ++;
    out << N << " " << N << std::endl;
    out.flush();

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    glColor3f(1,0,0);
    for (int n=0; n<nPlanes; ++n) {
      tdp::Plane& pl = pls.GetCircular(n);
      tdp::glDrawLine(pl.p_, pl.p_+scale*pl.n_);
    }

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();
    viewSeg.SetImage(z);
    viewCurv.SetImage(curv);

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
  out.close();
  return 0;
}


