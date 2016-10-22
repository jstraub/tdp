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

#include <tdp/gui/gui.hpp>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SO3.h>

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

  tdp::GUI gui(1200,800,video);

  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = w+w%64; // for convolution
  size_t hc = h+h%64;

  float f = 550;
  float uc = (w-1.)/2.;
  float vc = (h-1.)/2.;

  CameraT cam(Eigen::Vector4f(f,f,uc,vc)); 

  if (calibPath.size() > 0) {
    tdp::Rig<CameraT> rig;
    rig.FromFile(calibPath,false);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
    // camera model for computing point cloud and normals
    cam = rig.cams_[rig.rgbStream2cam_[0]];
  }

  tdp::QuickView viewN2D(wc,hc);
  gui.container().AddDisplay(viewN2D);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);

  tdp::QuickView viewGrey(wc,hc);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewGreyDu(wc,hc);
  gui.container().AddDisplay(viewGreyDu);
  tdp::QuickView viewGreyDv(wc,hc);
  gui.container().AddDisplay(viewGreyDv);
  tdp::QuickView viewGrad3Dimg(wc,hc);
  gui.container().AddDisplay(viewGrad3Dimg);

  pangolin::View& viewGrad3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewGrad3D);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<float> grey(wc, hc);
  tdp::ManagedHostImage<float> greydu(wc, hc);
  tdp::ManagedHostImage<float> greydv(wc, hc);

  tdp::ManagedDeviceImage<uint16_t> cuZ(wc,hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<tdp::Vector3fda> n2Df(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc,hc);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(w, h);
  tdp::ManagedDeviceImage<float> cuGrey(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreydu(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreydv(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuGrad3D(wc, hc);
  //tdp::ManagedDeviceImage<tdp::Vector3fda> cuGrad3Ddir(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuGrad3DdirImg(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3bda> grad3DdirImg(wc,hc);
  tdp::ManagedHostImage<tdp::Vector3fda> grad3Ddir(wc,hc);

  pangolin::GlBufferCudaPtr cuNbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  pangolin::GlBufferCudaPtr cuGrad3Dbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuDrawf(wc, hc);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::GeodesicHist<4> normalHist;
  tdp::GeodesicHist<4> grad3dHist;

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> tsdfDmin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> tsdfDmax("ui.d max",12.,0.1,16.);

  pangolin::Var<bool> verbose ("ui.verbose", false,true);
  pangolin::Var<bool>  compute3Dgrads("ui.compute3Dgrads",false,true);
  pangolin::Var<bool>  show2DNormals("ui.show 2D Normals",true,true);
  pangolin::Var<bool>  computeHist("ui.ComputeHist",true,true);
  pangolin::Var<bool>  histFrameByFrame("ui.hist frame2frame", false, true);
  pangolin::Var<float> histScale("ui.hist scale",40.,1.,100.);
  pangolin::Var<bool> histLogScale("ui.hist log scale",false,true);
  pangolin::Var<bool>  dispGrid("ui.Show Grid",false,true);
  pangolin::Var<bool>  dispNormals("ui.Show Normals",true,true);

  pangolin::Var<bool> dpvmfmeans("ui.DpvMFmeans", false,true);
  pangolin::Var<float> lambdaDeg("ui.lambdaDeg", 90., 1., 180.);
  pangolin::Var<int> maxIt("ui.max It", 10, 1, 100);
  pangolin::Var<float> minNchangePerc("ui.Min Nchange", 0.005, 0.001, 0.1);

  pangolin::Var<bool> runRtmf("ui.rtmf", true,true);
  pangolin::Var<float> tauR("ui.tau R", 10., 1., 100);

  pangolin::Var<float> gradNormThr("ui.grad norm thr", 6, 0, 10);

  tdp::vMFMMF<1> rtmf(w,h,tauR);

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

    TICK("Convert Depth");
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    ConvertDepthGpu(cuDraw, cuDrawf, depthSensorScale, tsdfDmin, tsdfDmax);
    tdp::Blur5(cuDrawf, cuD, 0.03);
    TOCK("Convert Depth");
    {
      TICK("Compute Normals");
      pangolin::CudaScopedMappedPtr cuNbufp(cuNbuf);
      cudaMemset(*cuNbufp,0, hc*wc*sizeof(tdp::Vector3fda));
      tdp::Image<tdp::Vector3fda> cuN(wc, hc,
          wc*sizeof(tdp::Vector3fda), (tdp::Vector3fda*)*cuNbufp);
      Depth2Normals(cuD, cam, tdp::SO3f(), cuN);
      n.CopyFrom(cuN,cudaMemcpyDeviceToHost);
      TOCK("Compute Normals");
      if (show2DNormals) {
        TICK("Compute 2D normals image");
        tdp::Normals2Image(cuN, cuN2D);
        n2D.CopyFrom(cuN2D,cudaMemcpyDeviceToHost);
        TOCK("Compute 2D normals image");
      }

      if (dpvmfmeans) {
        TICK("Compute DPvMFClustering");
        tdp::DPvMFmeans dpm(cos(lambdaDeg*M_PI/180.)); 
        dpm.Compute(n, cuN, cuZ, maxIt, minNchangePerc);
        TOCK("Compute DPvMFClustering");
      }
      if (runRtmf) {
        TICK("Compute RTMF");
        rtmf.Compute(cuN, maxIt, verbose);
        tdp::SO3f R_wc(rtmf.Rs_[0]);
        tdp::TransformPc(R_wc.Inverse(),cuN);
        TOCK("Compute RTMF");
      }

      if (computeHist) {
        TICK("Compute Hist");
        if (histFrameByFrame)
          normalHist.Reset();
        normalHist.ComputeGpu(cuN);
        TOCK("Compute Hist");
      }

      grad3Ddir.Fill(tdp::Vector3fda(0,0,1));
      pangolin::CudaScopedMappedPtr cuGrad3Dbufp(cuGrad3Dbuf);
      tdp::Image<tdp::Vector3fda> cuGrad3Ddir(wc, hc,
          wc*sizeof(tdp::Vector3fda), (tdp::Vector3fda*)*cuGrad3Dbufp);
      cuGrad3Ddir.CopyFrom(grad3Ddir,cudaMemcpyHostToDevice);

      cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
      tdp::Rgb2Grey(cuRgb,cuGrey);
      tdp::Gradient3D(cuGrey, cuD, cuN, cam, gradNormThr, cuGreydu,
          cuGreydv, cuGrad3D);
      cuGrad3Ddir.CopyFrom(cuGrad3D, cudaMemcpyDeviceToDevice);
      tdp::RenormalizeSurfaceNormals(cuGrad3Ddir, gradNormThr);
      tdp::Normals2Image(cuGrad3Ddir, cuGrad3DdirImg);

      grad3DdirImg.CopyFrom(cuGrad3DdirImg,cudaMemcpyDeviceToHost);
      grey.CopyFrom(cuGrey,cudaMemcpyDeviceToHost);
      greydu.CopyFrom(cuGreydu, cudaMemcpyDeviceToHost);
      greydv.CopyFrom(cuGreydv, cudaMemcpyDeviceToHost);

      if (runRtmf) {
        tdp::SO3f R_wc(rtmf.Rs_[0]);
        tdp::TransformPc(R_wc.Inverse(),cuGrad3Ddir);
      }

      if (histFrameByFrame)
        grad3dHist.Reset();
      grad3dHist.ComputeGpu(cuGrad3Ddir);
    }

    tdp::SE3f T_wc;
    if (runRtmf) {
      T_wc.rotation() = tdp::SO3f(rtmf.Rs_[0]);
    }

    TICK("Render 3D");
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    glLineWidth(1.5f);
    pangolin::glDrawAxis(1);
    glColor4f(0,1,0,0.5);
    if (dispNormals) {
//      pangolin::glSetFrameOfReference(T_wc.matrix());
      pangolin::RenderVbo(cuNbuf);
//      pangolin::glUnsetFrameOfReference();
    }
    if (computeHist) {
      if (dispGrid) {
        normalHist.geoGrid_.Render3D();
      }
      normalHist.Render3D(histScale, histLogScale);
    }

    viewGrad3D.Activate(s_cam);
    glLineWidth(1.5f);
    pangolin::glDrawAxis(1);
    glColor4f(0,1,0,0.5);
    pangolin::RenderVbo(cuGrad3Dbuf);
    if (dispGrid) {
      grad3dHist.geoGrid_.Render3D();
    }
    grad3dHist.Render3D(histScale, histLogScale);
    TOCK("Render 3D");

    TICK("Render 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    gui.ShowFrames();

    if (show2DNormals) {
      viewN2D.SetImage(n2D);
    }
    viewGrad3Dimg.SetImage(grad3DdirImg);
    viewGrey.SetImage(grey);
    viewGreyDu.SetImage(greydu);
    viewGreyDv.SetImage(greydv);
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
