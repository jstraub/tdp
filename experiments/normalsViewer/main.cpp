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

#include <tdp/gui/gui.hpp>
#include <tdp/camera/rig.h>
#include <tdp/manifold/SO3.h>

#include <tdp/distributions/vmf_mm.h>

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

template <typename T>
Eigen::Matrix<T,4,4> BuildM(const
    Eigen::Matrix<T,3,1>& u, const Eigen::Matrix<T,3,1>& v) {
  const T ui = u(0);
  const T uj = u(1);
  const T uk = u(2);
  const T vi = v(0);
  const T vj = v(1);
  const T vk = v(2);
  Eigen::Matrix<T,4,4> M;
  M << u.transpose()*v, uk*vj-uj*vk,       ui*vk-uk*vi,       uj*vi-ui*vj, 
       uk*vj-uj*vk,     ui*vi-uj*vj-uk*vk, uj*vi+ui*vj,       ui*vk+uk*vi,
       ui*vk-uk*vi,     uj*vi+ui*vj,       uj*vj-ui*vi-uk*vk, uj*vk+uk*vj,
       uj*vi-ui*vj,     ui*vk+uk*vi,       uj*vk+uk*vj,       uk*vk-ui*vi-uj*vj;
  return M;
}

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
  pangolin::View& viewNormals3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewNormals3D);
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);
  pangolin::View& viewGrad3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewGrad3D);

  tdp::QuickView viewN2D(wc,hc);
  gui.container().AddDisplay(viewN2D);
  tdp::QuickView viewGrey(wc,hc);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewGreyDu(wc,hc);
  gui.container().AddDisplay(viewGreyDu);
  tdp::QuickView viewGreyDv(wc,hc);
  gui.container().AddDisplay(viewGreyDv);
  tdp::QuickView viewGrad3Dimg(wc,hc);
  gui.container().AddDisplay(viewGrad3Dimg);
  tdp::QuickView viewZ(wc,hc);
  gui.container().AddDisplay(viewZ);

  viewN2D.Show(true);
  viewGrey.Show(false);
  viewGreyDu.Show(false);
  viewGreyDv.Show(false);
  viewGrad3Dimg.Show(false);
  viewGrad3D.Show(false);

  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
  pangolin::DataLog logF;
  pangolin::Plotter plotF(&logF, -100.f,1.f, 0.f,40.f, 10.f, 0.1f);
  plotters.AddDisplay(plotF);
  pangolin::DataLog logEig;
  pangolin::Plotter plotEig(&logEig, -100.f,1.f, -0.f,1.3f, 10.f, 0.1f);
  plotters.AddDisplay(plotEig);
  gui.container().AddDisplay(plotters);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<float> grey(wc, hc);
  tdp::ManagedHostImage<float> greydu(wc, hc);
  tdp::ManagedHostImage<float> greydv(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPc(wc, hc);

  tdp::ManagedHostImage<uint16_t> z(wc,hc);
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

  tdp::ManagedDeviceImage<tdp::Vector3fda> cuDirs(wc, 2*hc);

  pangolin::GlBufferCudaPtr cuNbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  pangolin::GlBufferCudaPtr cuGrad3Dbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

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
  pangolin::Var<bool>  computeHist("ui.ComputeHist",true,true);
  pangolin::Var<bool>  histFrameByFrame("ui.hist frame2frame", false, true);
  pangolin::Var<float> histScale("ui.hist scale",40.,1.,100.);
  pangolin::Var<bool> histLogScale("ui.hist log scale",false,true);
  pangolin::Var<bool>  dispGrid("ui.Show Grid",false,true);
  pangolin::Var<bool>  dispNormals("ui.Show Normals",true,true);

  pangolin::Var<bool> dpvmfmeans("ui.DpvMFmeans", false,true);
  pangolin::Var<float> lambdaDeg("ui.lambdaDeg", 65., 1., 180.);
  pangolin::Var<int> maxIt("ui.max It", 10, 1, 100);
  pangolin::Var<float> minNchangePerc("ui.Min Nchange", 0.005, 0.001, 0.1);

  pangolin::Var<bool> runRtmf("ui.rtmf", false,true);
  pangolin::Var<float> tauR("ui.tau R", 10., 1., 100);

  pangolin::Var<float> gradNormThr("ui.grad norm thr", 6, 0, 10);

  pangolin::Var<bool> addGrad3("ui.add Grad3d", true,true);
  pangolin::Var<bool> runNormals2vMF("ui.normals2vMF", true,true);
  pangolin::Var<float> kfThr("ui.KF thr", 0.9, 0.5, 1.0);
  pangolin::Var<bool> newKf("ui.new KF", true,false);
  pangolin::Var<bool> filterHalfSphere("ui.filter half sphere", true, true);

  tdp::vMFMMF<1> rtmf(tauR);
  std::vector<tdp::vMF<float,3>> vmfs;
  Eigen::Matrix<float,4,Eigen::Dynamic> xSums;

  tdp::SO3fda R_cvMF;
  tdp::SO3fda R_wc;
  tdp::SO3fda R_cw;

  size_t nFramesTracked = 0;
  float f = 1.;
  float fKF = 1.;

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

    TICK("Compute grey");
    cuRgb.CopyFrom(rgb);
    tdp::Rgb2Grey(cuRgb,cuGrey);
    grey.CopyFrom(cuGrey);
    TOCK("Compute grey");


    TICK("Convert Depth");
    cuDraw.CopyFrom(dRaw);
    ConvertDepthGpu(cuDraw, cuDrawf, depthSensorScale, tsdfDmin, tsdfDmax);
    tdp::Blur5(cuDrawf, cuD, 0.03);
    TOCK("Convert Depth");
    {
      TICK("Compute Normals");
      pangolin::CudaScopedMappedPtr cuNbufp(cuNbuf);
      cudaMemset(*cuNbufp,0, hc*wc*sizeof(tdp::Vector3fda));
      tdp::Image<tdp::Vector3fda> cuN(wc, hc,
          wc*sizeof(tdp::Vector3fda), (tdp::Vector3fda*)*cuNbufp,
          tdp::Storage::Gpu);
      Depth2Normals(cuD, cam, tdp::SE3f(), cuN);
      n.CopyFrom(cuN);
      TOCK("Compute Normals");

      TICK("Compute 3D grads");
      grad3Ddir.Fill(tdp::Vector3fda(0,0,1));
      pangolin::CudaScopedMappedPtr cuGrad3Dbufp(cuGrad3Dbuf);
      tdp::Image<tdp::Vector3fda> cuGrad3Ddir(wc, hc,
          wc*sizeof(tdp::Vector3fda), (tdp::Vector3fda*)*cuGrad3Dbufp,
          tdp::Storage::Gpu);
      cuGrad3Ddir.CopyFrom(grad3Ddir);

      tdp::Gradient3D(cuGrey, cuD, cuN, cam, gradNormThr, cuGreydu,
          cuGreydv, cuGrad3D);
      cuGrad3Ddir.CopyFrom(cuGrad3D);
      tdp::RenormalizeSurfaceNormals(cuGrad3Ddir, gradNormThr);
      tdp::Normals2Image(cuGrad3Ddir, cuGrad3DdirImg);

      grad3DdirImg.CopyFrom(cuGrad3DdirImg);
      greydu.CopyFrom(cuGreydu);
      greydv.CopyFrom(cuGreydv);
      TOCK("Compute 3D grads");

      if (viewN2D.IsShown()) { 
        TICK("Compute 2D normals image");
        tdp::Normals2Image(cuN, cuN2D);
        n2D.CopyFrom(cuN2D);
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

      if (addGrad3) {
        cuDirs.Reinitialise(wc,hc*2);
        cuDirs.GetRoi(0,0,wc,hc).CopyFrom(cuN);
        cuDirs.GetRoi(0,hc,wc,hc).CopyFrom(cuGrad3Ddir);
      } else {
        cuDirs.Reinitialise(wc,hc);
        cuDirs.CopyFrom(cuN);
      }

      if (runNormals2vMF && (pangolin::Pushed(newKf) || f/fKF < kfThr )) { 
        tdp::DPvMFmeans dpm(cos(lambdaDeg*M_PI/180.)); 
        tdp::ComputevMFMM(n, cuDirs, dpm, maxIt, minNchangePerc, 
            cuZ, vmfs);
//        R_cw = R_cw * R_cvMF;
        R_cw = R_cvMF * R_cw;
        R_cvMF = tdp::SO3fda();
        nFramesTracked = 0;
      }
      if (runNormals2vMF) {
        if (nFramesTracked > 0)
          tdp::MAPLabelAssignvMFMM(vmfs, R_cvMF, cuDirs,  cuZ, filterHalfSphere);
        xSums = tdp::SufficientStats1stOrder(cuDirs, cuZ, vmfs.size());
        Eigen::Matrix3d N = Eigen::Matrix3d::Zero();
        for (size_t k=0; k<vmfs.size(); ++k) {
          //TODO: push this into the writeup!
          N += vmfs[k].GetTau()
            *xSums.block<3,1>(0,k).cast<double>()
            *vmfs[k].GetMu().cast<double>().transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(N,
            Eigen::ComputeFullU|Eigen::ComputeFullV);
        double sign = (svd.matrixU()*svd.matrixV().transpose()).determinant();
        Eigen::Matrix3Xd dR = svd.matrixU()
          *Eigen::Vector3d(1.,1.,sign).asDiagonal()*svd.matrixV().transpose();
//        std::cout << dR << std::endl;
        std::cout << " fit: " << (N*dR).trace()  <<  " " 
          << (N*dR).trace()/xSums.bottomRows<1>().sum()
          << " singular values " << svd.singularValues().transpose()
          << std::endl;
//        R_cvMF = tdp::SO3fda(dR.cast<float>()) * R_cvMF;
        R_cvMF = tdp::SO3fda(dR.cast<float>());
//        R_cvMF = R_cvMF * tdp::SO3fda(dR);
//
        f = (N*dR).trace()/xSums.bottomRows<1>().sum();
        
        Eigen::Matrix4f M = Eigen::Matrix4f::Zero();
        for (size_t k=0; k<vmfs.size(); ++k) {
          Eigen::Vector3f xSum = xSums.block<3,1>(0,k);
          M += vmfs[k].GetTau()*BuildM(xSum,vmfs[k].GetMu());
        }
//        std::cout << M << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> eig(M);
        Eigen::Vector4f e = eig.eigenvalues()/eig.eigenvalues()(0);
        std::cout << e.transpose() << std::endl;

//        logEig.Log(e(0), e(1), e(2), e(3));
        if (nFramesTracked == 0) { 
          fKF = f;
        }
        logF.Log(f, R_cvMF.Log().norm()*180./M_PI);
        logEig.Log(e.array().prod(), kfThr, f/fKF);
        nFramesTracked ++;
      }

      if (computeHist) {
        TICK("Compute Hist");
        if (histFrameByFrame)
          normalHist.Reset();
        normalHist.ComputeGpu(cuN);
        TOCK("Compute Hist");
      }


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
      T_wc.rotation() = rtmf.Rs_[0];
    }

    tdp::SO3f R_wc = (R_cvMF*R_cw).Inverse();
    tdp::SO3f R_wcPrev = (R_cw).Inverse();

    TICK("Render 3D");
    glEnable(GL_DEPTH_TEST);
    if (viewNormals3D.IsShown()) {
      viewNormals3D.Activate(s_cam);
      glLineWidth(1.5f);
      pangolin::glDrawAxis(1);
      glColor4f(0,1,0,0.5);
      tdp::SE3fda T_wc(R_wc);
      if (dispNormals) {
        //      pangolin::glSetFrameOfReference(T_wc.matrix());
        pangolin::glSetFrameOfReference(T_wc.matrix());
        pangolin::RenderVbo(cuNbuf);
        pangolin::glUnsetFrameOfReference();
        //      pangolin::glUnsetFrameOfReference();
      }
      if (runNormals2vMF) {
        glColor4f(0,1,1,1);
        tdp::SE3fda T_wcPrev(R_wcPrev);
        pangolin::glSetFrameOfReference(T_wcPrev.matrix());
        for (const auto& vmf : vmfs) {
          tdp::glDrawLine(Eigen::Vector3f::Zero(), vmf.GetMu());
        }
        pangolin::glUnsetFrameOfReference();
        glColor4f(1,1,0,1);
        pangolin::glSetFrameOfReference(T_wc.matrix());
        for (size_t k=0; k<xSums.cols(); ++k) {
          Eigen::Vector3f dir = xSums.block<3,1>(0,k).normalized();
          tdp::glDrawLine(Eigen::Vector3f::Zero(), dir);
        }
        pangolin::glUnsetFrameOfReference();
      }
      if (computeHist) {
        pangolin::glSetFrameOfReference(T_wc.matrix());
        if (dispGrid) {
          normalHist.geoGrid_.Render3D();
        }
        normalHist.Render3D(histScale, histLogScale);
        pangolin::glUnsetFrameOfReference();
      }
    }
    if (viewPc3D.IsShown()) {
      viewPc3D.Activate(s_cam);
      if (runNormals2vMF) {
        tdp::SE3fda T_wc(R_wc);
        pangolin::glSetFrameOfReference(T_wc.matrix());
      }
      tdp::Depth2PCGpu(cuD,cam,cuPc);
      pc.CopyFrom(cuPc);
      vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);
      cbo.Upload(rgb.ptr_, rgb.SizeBytes(), 0);
      pangolin::RenderVboCbo(vbo, cbo);
      if (runNormals2vMF) {
        pangolin::glUnsetFrameOfReference();
      }
    }

    if (viewGrad3D.IsShown()) {
      viewGrad3D.Activate(s_cam);
      glLineWidth(1.5f);
      pangolin::glDrawAxis(1);
      glColor4f(0,1,0,0.5);
      pangolin::RenderVbo(cuGrad3Dbuf);
      if (dispGrid) {
        grad3dHist.geoGrid_.Render3D();
      }
      grad3dHist.Render3D(histScale, histLogScale);
    }

    TOCK("Render 3D");

    TICK("Render 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);

    if (viewZ.IsShown()) {
      z.CopyFrom(cuZ);
      for (size_t i=0; i<z.Area(); ++i)
        z[i] = std::min(z[i],(uint16_t)vmfs.size());
      viewZ.SetImage(z);
    }
    if (viewN2D.IsShown()) viewN2D.SetImage(n2D);
    if (viewGrad3Dimg.IsShown()) viewGrad3Dimg.SetImage(grad3DdirImg);
    if (viewGrey.IsShown()) viewGrey.SetImage(grey);
    if (viewGreyDu.IsShown()) viewGreyDu.SetImage(greydu);
    if (viewGreyDv.IsShown()) viewGreyDv.SetImage(greydv);
    plotF.ScrollView(1,0);
    plotEig.ScrollView(1,0);
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
