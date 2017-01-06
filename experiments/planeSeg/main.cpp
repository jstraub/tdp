#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <tdp/preproc/convolutionSeparable.h>
#include <tdp/preproc/depth.h>
#include <tdp/preproc/normals.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/gui/quickView.h>
#include <tdp/directional/hist.h>
#include <tdp/clustering/dpvmfmeans.hpp>
#include <tdp/rtmf/vMFMF.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/preproc/blur.h>
#include <tdp/manifold/SE3.h>
#include <tdp/preproc/pc.h>
#include <tdp/preproc/grad.h>
#include <tdp/preproc/grey.h>
#include <tdp/preproc/grey.h>
#include <tdp/preproc/project.h>
#include <tdp/preproc/plane.h>
#include <tdp/gl/render.h>
#include <tdp/clustering/dpvmfmeans_simple.hpp>

#include <tdp/gui/gui.hpp>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SO3.h>

#include <tdp/distributions/vmf_mm.h>

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

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

  tdp::GuiBase gui(1200,800,video);

  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = w+w%64; // for convolution
  size_t hc = h+h%64;

  float uc = (w-1.)/2.;
  float vc = (h-1.)/2.;

  CameraT cam(Eigen::Vector4f(550,550,uc,vc)); 

  if (calibPath.size() > 0) {
    tdp::Rig<CameraT> rig;
    rig.FromFile(calibPath,false);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
    // camera model for computing point cloud and normals
    cam = rig.cams_[rig.rgbStream2cam_[0]];
  }

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);

  tdp::QuickView viewN2D(wc,hc);
  gui.container().AddDisplay(viewN2D);
  tdp::QuickView viewGrey(wc,hc);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewZ(wc,hc);
  gui.container().AddDisplay(viewZ);
  tdp::QuickView viewProj(wc,hc);
  gui.container().AddDisplay(viewProj);

  viewN2D.Show(true);
  viewGrey.Show(false);

//  pangolin::View& plotters = pangolin::Display("plotters");
//  plotters.SetLayout(pangolin::LayoutEqualVertical);
//  pangolin::DataLog logF;
//  pangolin::Plotter plotF(&logF, -100.f,1.f, 0.f,40.f, 10.f, 0.1f);
//  plotters.AddDisplay(plotF);
//  pangolin::DataLog logEig;
//  pangolin::Plotter plotEig(&logEig, -100.f,1.f, -0.f,1.3f, 10.f, 0.1f);
//  plotters.AddDisplay(plotEig);
//  gui.container().AddDisplay(plotters);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<float> grey(wc, hc);
  tdp::ManagedHostImage<float> greydu(wc, hc);
  tdp::ManagedHostImage<float> greydv(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPc(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(wc, hc);

  tdp::ManagedHostImage<tdp::Vector4fda> pl(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector4fda> cuPl(wc, hc);

  tdp::ManagedHostImage<float> proj(wc, hc);
  tdp::ManagedDeviceImage<float> cuProj(wc, hc);

  tdp::ManagedHostImage<uint16_t> z(wc,hc);
  tdp::ManagedDeviceImage<uint16_t> cuZ(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(w, h);
  tdp::ManagedDeviceImage<float> cuGrey(wc, hc);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);
  pangolin::GlBuffer lbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_SHORT,1);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuDrawf(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> tsdfDmin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> tsdfDmax("ui.d max",12.,0.1,16.);

  pangolin::Var<bool>  verbose ("ui.verbose", false,true);
  pangolin::Var<bool>  dispNormals("ui.Show Normals",true,true);
  pangolin::Var<bool>  showLabels("ui.Show Labels",true,true);

  pangolin::Var<float> lambdaDeg("ui.lambdaDeg", 65., 1., 180.);
  pangolin::Var<int>   maxIt("ui.max It", 100, 1, 100);
  pangolin::Var<float> eps("ui.eps", 0.005, 0.001, 0.1);

  pangolin::Var<float> blurThr("ui.blur Thr", 0.03, 0.01, 0.2);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    TICK("Read frame");
    gui.NextFrames();
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    TOCK("Read frame");
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGrey);

    TICK("Convert Depth");
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    ConvertDepthGpu(cuDraw, cuDrawf, depthSensorScale, tsdfDmin, tsdfDmax);
    tdp::Blur9(cuDrawf, cuD, 1.2*blurThr);
    tdp::Blur9(cuD, cuDrawf, blurThr);
    tdp::Blur9(cuDrawf, cuD, 0.8*blurThr);
    TOCK("Convert Depth");
    TICK("Compute Normals");
    Depth2Normals(cuD, cam, tdp::SE3f(), cuN);
    n.CopyFrom(cuN,cudaMemcpyDeviceToHost);
    TOCK("Compute Normals");

    TICK("Compute Planes");
    tdp::ComputeUnitPlanes(cuPc, cuN, cuPl);
    pl.CopyFrom(cuPl);
    tdp::ProjectPc(cuPc, cuN, cuProj);
    proj.CopyFrom(cuProj);
    TOCK("Compute Normals");

    TICK("Compute DPvMFClustering");
    tdp::DPvMFmeansSimple<float,4,Eigen::DontAlign> dpvmf(cos(lambdaDeg*M_PI/180.)); 
    for (size_t i=0; i<pl.Area(); ++i) {
      if (tdp::IsValidData(pl[i])) {
        dpvmf.addObservation(&pl[i], &z[i]); 
      }
    }
    dpvmf.iterateToConvergence(maxIt, eps);
    TOCK("Compute DPvMFClustering");
//    z.Fill(0);
    size_t j=0;
    for (size_t i=0; i<pl.Area(); ++i) {
      if (tdp::IsValidData(pl[i])) {
//        z[i] = (*dpvmf.GetZs()[j++])+1;
        z[i] ++;
      }
    }

//    tdp::MAPLabelAssignvMFMM(vmfs, R_cvMF, cuN,  cuZ, filterHalfSphere);
//    xSums = tdp::SufficientStats1stOrder(cuN, cuZ, vmfs.size());

    TICK("Render 3D");
    glEnable(GL_DEPTH_TEST);
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);
      tdp::Depth2PCGpu(cuD,cam,cuPc);
      pc.CopyFrom(cuPc);
      vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);

      if (showLabels) {
        lbo.Upload(z.ptr_, z.SizeBytes(), 0);
        tdp::RenderLabeledVbo(vbo, lbo, s_cam, dpvmf.GetK()+1);
      } else {
        cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
        pangolin::RenderVboCbo(vbo, cbo);
      }
    }
    TOCK("Render 3D");

    TICK("Render 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (viewZ.IsShown()) {
      viewZ.SetImage(z);
    }
    if (viewN2D.IsShown()) {
      tdp::Normals2Image(cuN, cuN2D);
      n2D.CopyFrom(cuN2D);
      viewN2D.SetImage(n2D);
    }
    if (viewGrey.IsShown()) viewGrey.SetImage(grey);
    if (viewProj.IsShown()) viewProj.SetImage(proj);
    TOCK("Render 2D");

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }
  return 0;
}
