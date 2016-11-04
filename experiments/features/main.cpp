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
#include <tdp/preproc/normals.h>

#include <tdp/gui/gui.hpp>
#include <tdp/features/fast.h>
#include <tdp/features/brief.h>
#include <tdp/preproc/grey.h>
#include <tdp/preproc/blur.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/registration/robust3D3D.h>
#include <tdp/ransac/ransac.h>

#include <tdp/data/pyramid.h>
#include <tdp/camera/photometric.h>
#include <tdp/camera/rig.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/icp/icp.h>
#include <tdp/features/lsh.h>

typedef tdp::CameraPoly3f CameraT;


namespace tdp {
struct BinaryKF {
  BinaryKF(size_t w, size_t h)
    : pyrGrey(w,h), pyrPc(w,h), lsh(11)
  {}

  ManagedHostImage<uint8_t> pyrGrey;
  ManagedHostPyramid<Vector3fda,3> pyrPc;
  ManagedHostImage<Brief> feats;
  LshForest<14> lsh;

};

void MatchKFs(const std::vector<BinaryKF>& kfs, int briefMatchThr,
    int ransacMaxIt, float ransacThr) {

  auto& kfA = kfs[kfs.size()-1];
  for (size_t i=0; i<kfs.size()-1; ++i) {
    auto& kfB = kfs[i];
    kfB.lsh.PrintFillStatus();

    TICK("MatchKFs");
    std::vector<uint32_t> assoc;
    std::vector<Vector3fda> featB;
    assoc.reserve(kfA.feats.w_);
    featB.reserve(kfA.feats.w_);
    for (size_t j=0; j<kfA.feats.w_; ++j) {
      Brief feat;
      int dist;
      if (kfB.lsh.SearchBest(kfA.feats(j,1),dist,feat)
          && dist < briefMatchThr) {
        std::cout << dist << " ";
        assoc.push_back(j);
        featB.push_back(feat.p_c_);
      }
    }
    TOCK("MatchKFs");
    std::cout << std::endl;
    std::cout << kfs.size()-1 <<  " -> " << i << ": " 
      << assoc.size()/(float)kfA.feats.Area()
      << std::endl;

    TICK("RANSAC");
    ManagedHostImage<Vector3fda> pcA(assoc.size());
    ManagedHostImage<Vector2ida> assocAB(assoc.size());
    Image<Vector3fda> pcB(featB.size(), 1, &featB[0]);
    for (size_t i=0; i<assoc.size(); ++i) {
      pcA[i] = kfA.feats(assoc[i],1).p_c_;
      assocAB[i](0) = i;
      assocAB[i](1) = i;
    }
    P3P p3p;
    Ransac<Vector3fda> ransac(&p3p);
    size_t numInliers = 0;
    SE3f T_ab = ransac.Compute(pcA, pcB, assocAB, ransacMaxIt,
        ransacThr, numInliers);
    TOCK("RANSAC");
    
    std::cout << kfs.size()-1 <<  " -> " << i << ": " 
      << assoc.size() << " " << assoc.size()/(float)kfA.feats.Area() 
      << " after RANSAC "
      << numInliers << " " << numInliers/(float)kfA.feats.Area() 
      << std::endl;
  }
}

}

int main( int argc, char* argv[] )
{
  std::string input_uri = "openni2://";
  std::string output_uri = "pango://video.pango";
  std::string calibPath = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";
  bool runOnce = false;

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
    if (argc > 3 && std::string(argv[3]).compare("-1") == 0 ) runOnce = true;
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

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  tdp::Rig<CameraT> rig;
  if (calibPath.size() > 0) {
    rig.FromFile(calibPath,false);
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    rig.CorrespondOpenniStreams2Cams(streams);
  } else {
    return 2;
  }

  tdp::GUI gui(1200,800,video);

  size_t w = video.Streams()[gui.iD[0]].Width();
  size_t h = video.Streams()[gui.iD[0]].Height();
  size_t wOrig = video.Streams()[gui.iD[0]].Width();
  size_t hOrig = video.Streams()[gui.iD[0]].Height();
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  size_t wc = (w+w%64); // for convolution
  size_t hc = rig.NumCams()*(h+h%64);
  w = wc;
  h = hc;

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
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);
  tdp::QuickView viewGrey(w,h);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewAssoc(w*2,h);
  gui.container().AddDisplay(viewAssoc);

  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
  pangolin::DataLog logInliers;
  pangolin::Plotter plotInliers(&logInliers, -100.f,1.f, 0, 1.f, 
      1.f, 0.1f);
  plotters.AddDisplay(plotInliers);
  pangolin::DataLog logOverlap;
  pangolin::Plotter plotOverlap(&logOverlap, -100.f,1.f, 0.f,1.f, 1.f, 0.1f);
  plotters.AddDisplay(plotOverlap);
  pangolin::DataLog logRmse;
  pangolin::Plotter plotRmse(&logRmse, -100.f,1.f, 0.f, 100.0f, 0.1f, 0.1f);
  plotters.AddDisplay(plotRmse);
  pangolin::DataLog logdH;
  pangolin::Plotter plotdH(&logdH, -100.f,1.f, 0.7f, 1.3f, 0.1f, 0.1f);
  plotters.AddDisplay(plotdH);
  gui.container().AddDisplay(plotters);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<uint8_t> grey(wOrig, hOrig);
  tdp::ManagedHostImage<uint8_t> greyB(wOrig, hOrig);
  tdp::ManagedHostImage<uint8_t> greyAssoc(wOrig*2, hOrig);
  tdp::ManagedDevicePyramid<uint8_t,3> cuPyrGrey(wOrig, hOrig);
  tdp::ManagedHostPyramid<uint8_t,3> pyrGrey(wOrig, hOrig);

  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pcB(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wOrig, hOrig);
  tdp::ManagedDeviceImage<float> cuGreyOrig(wOrig, hOrig);
  tdp::ManagedDeviceImage<float> cuGrey(wc, hc);
  tdp::ManagedDeviceImage<uint8_t> cuGreyChar(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreydu(wc, hc);
  tdp::ManagedDeviceImage<float> cuGreydv(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuGrad3D(wc, hc);


  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);

  tdp::ManagedHostPyramid<tdp::Vector3fda,3> pyrPc(wc,hc);

  tdp::ManagedHostImage<tdp::Vector3fda> grad3D(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc, hc);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> useHuber("ui.Use Huber", false, true);
  pangolin::Var<float> huberDelta("ui.huber Delta",0.1,0.01,1.);

  pangolin::Var<bool> useRansac("ui.Use RANSAC", true, true);
  pangolin::Var<bool> runICP("ui.run ICP", true, true);
  pangolin::Var<float> ransacMaxIt("ui.max it",3000,1,1000);
  pangolin::Var<float> ransacThr("ui.thr",0.03,0.01,1.0);
  pangolin::Var<float> ransacInlierPercThr("ui.inlier % thr",0.15,0.1,1.0);

  pangolin::Var<int> fastB("ui.FAST b",30,0,100);
  pangolin::Var<float> harrisThr("ui.harris thr",0.1,0.001,2.0);
  pangolin::Var<float> kappaHarris("ui.kappa harris",0.08,0.04,0.15);
  pangolin::Var<int> briefMatchThr("ui.BRIEF match",65,0,100);
  pangolin::Var<bool> newKf("ui.new KF", false, false);
  pangolin::Var<float> dEntropyThr("ui.dH Thr",0.9,0.8,1.0);

  pangolin::Var<float> icpLoopCloseAngleThr_deg("ui.icpLoop angle thr",20,0.,90.);
  pangolin::Var<float> icpLoopCloseDistThr("ui.icpLoop dist thr",0.30,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  tdp::ManagedHostImage<tdp::Brief> descsA;
  tdp::ManagedHostImage<tdp::Brief> descsB;

  tdp::ManagedHostImage<tdp::Vector2ida> ptsA;
  tdp::ManagedHostImage<float> orientations;
  tdp::ManagedHostImage<float> orientationsB;
  tdp::ManagedHostImage<tdp::Vector2ida> ptsB;

  tdp::ManagedHostImage<tdp::SE3f> cosys;

  tdp::ManagedHostImage<int32_t> assoc;

  tdp::ManagedHostImage<tdp::Vector2ida> assocAB;

  gui.verbose = false;

  bool updatedEntropy = false;
  float dH = 0.f;
  float dHkf = 0.;

  std::vector<tdp::BinaryKF> kfs;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (gui.frame == 1 || pangolin::Pushed(newKf)
        || dH/dHkf < dEntropyThr) {
      if (gui.verbose) std::cout << "kf" << std::endl;
      descsB.Reinitialise(descsA.w_, descsA.h_);
      descsB.CopyFrom(descsA, cudaMemcpyHostToHost);
      ptsB.Reinitialise(ptsA.w_, 1);
      ptsB.CopyFrom(ptsA, cudaMemcpyHostToHost);
      greyB.CopyFrom(grey, cudaMemcpyHostToHost);
      pcB.CopyFrom(pc, cudaMemcpyHostToHost);
      orientationsB.Reinitialise(ptsA.w_,1);
      orientationsB.CopyFrom(orientations, cudaMemcpyHostToHost);

      pcs_m.CopyFrom(pcs_c, cudaMemcpyDeviceToHost);
      ns_m.CopyFrom(ns_c, cudaMemcpyDeviceToDevice);
      updatedEntropy = true;

      kfs.emplace_back(wc,hc);
      kfs.back().pyrPc.CopyFrom(pcs_c, cudaMemcpyDeviceToHost);
      kfs.back().pyrGrey.CopyFrom(grey, cudaMemcpyHostToHost);
      kfs.back().lsh.Insert(descsA);
      kfs.back().feats.Reinitialise(descsA.w_, descsA.h_);
      kfs.back().feats.CopyFrom(descsA, cudaMemcpyHostToHost);

      tdp::MatchKFs(kfs, briefMatchThr, ransacMaxIt, ransacThr);
    } else if (kfs.size() > 0) {
      kfs.back().lsh.Insert(descsA);
    }

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    if (gui.verbose) std::cout << "rgb" << std::endl;
    // get rgb image
//    rig.CollectRGB(gui, rgb, cudaMemcpyHostToHost) ;
//    tdp::Image<tdp::Vector2fda> cuGradGrey_c = cuPyrGradGrey_c.GetImage(0);
//    tdp::Gradient(cuGrey, cuGreyDu, cuGreyDv, cuGradGrey_c);
//    greyDu.CopyFrom(cuGreyDu, cudaMemcpyDeviceToHost);
//    greyDv.CopyFrom(cuGreyDv, cudaMemcpyDeviceToHost);
//    tdp::ConstructPyramidFromImage(cuGrey, cuPyrGrey_c,
//        cudaMemcpyDeviceToDevice);
//    tdp::CompletePyramid(cuPyrGradGrey_c, cudaMemcpyDeviceToDevice);
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGreyOrig,1.);
    tdp::Image<uint8_t> cuGrey0 = cuPyrGrey.GetImage(0);
    tdp::Blur5(cuGreyOrig,cuGrey0, 10.);
    tdp::CompletePyramid(cuPyrGrey, cudaMemcpyDeviceToDevice);

    // get depth image
//    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    if (gui.verbose) std::cout << "depth" << std::endl;
    int64_t t_host_us_d =0;
    rig.CollectD(gui, dMin, dMax, cuDraw, cuD, t_host_us_d);
    rig.ComputePc(cuD, true, pcs_c);
    rig.ComputeNormals(cuD, true, ns_c);

    // copy raw image to gpu
//    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    // convet depth image from uint16_t to float [m]
//    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
//    // compute point cloud (on CPU)
//    tdp::Depth2PC(d,cam,pc);
//    // compute normals
//    tdp::Depth2Normals(cuD, cam, cuN);

    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
    pc.CopyFrom(pcs_c.GetImage(0), cudaMemcpyDeviceToHost);
    cuN.CopyFrom(ns_c.GetImage(0), cudaMemcpyDeviceToDevice);
    pyrPc.CopyFrom(pcs_c, cudaMemcpyDeviceToHost);

    int fastLvl = 0;
//    tdp::Blur9(cuGrey,cuGreyChar, 10.);
    grey.CopyFrom(cuPyrGrey.GetImage(fastLvl), cudaMemcpyDeviceToHost);
    pyrGrey.CopyFrom(cuPyrGrey, cudaMemcpyDeviceToHost);
    if (gui.verbose) std::cout << "detect" << std::endl;
    TICK("Detection");
    tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, 18, ptsA, orientations);
    TOCK("Detection");

    if (gui.verbose) std::cout << "extract" << std::endl;
    TICK("Extraction");
//    tdp::ExtractBrief(grey, ptsA, orientations, descsA);
    tdp::ExtractBrief(pyrGrey, ptsA, orientations, gui.frame, fastLvl, descsA);
    for (size_t i=0; i<descsA.Area(); ++i) {
      descsA[i].p_c_ = pyrPc(descsA[i].lvl_, descsA[i].pt_(0), descsA[i].pt_(1));
    }
    TOCK("Extraction");


//    tdp::Gradient3D(cuGrey, cuD, cuN, cam, 0.001f, cuGreydu,
//        cuGreydv, cuGrad3D);
//    grad3D.CopyFrom(cuGrad3D, cudaMemcpyDeviceToHost);
//    n.CopyFrom(cuN, cudaMemcpyDeviceToHost);
//
//    cosys.Reinitialise(ptsA.w_,1);
//    for (size_t i=0; i<ptsA.Area(); ++i) {
//      const tdp::Vector3fda& pci = pc(ptsA[i](0),ptsA[i](1));
//      const tdp::Vector3fda& ni = n(ptsA[i](0),ptsA[i](1));
//      tdp::Vector3fda gradi = grad3D(ptsA[i](0),ptsA[i](1)).normalized();
//      tdp::SO3f R = tdp::SO3f::FromOrthogonalVectors(ni, gradi);
//      cosys[i] = tdp::SE3f(R, pci);
//    }

    if (gui.verbose) std::cout << "matching" << std::endl;
    TICK("Matching");
    int numMatches = 0;
    assoc.Reinitialise(descsA.w_,1);
    for (size_t i=0; i<descsA.w_; ++i) {
      int dist = 256;
      // match from current level 1 to all other levels
      assoc[i] = tdp::ClosestBrief(descsA(i,0), descsB, &dist);
      if (dist >= briefMatchThr ) {
//          || !tdp::IsValidData(pc(ptsA[i](0),ptsA[i](1)))
//          || !tdp::IsValidData(pcB(ptsB[assoc[i]](0),ptsB[assoc[i]](1)))) {
        assoc[i] = -1; 
      } else {
        numMatches ++;
      }

    }
    int32_t j=0;
    assocAB.Reinitialise(numMatches,1);
    for (size_t i=0; i<assoc.Area(); ++i) {
      if (assoc[i] >= 0) {
        assocAB[j](0) = ptsA[i](0)+ptsA[i](1)*w;
        assocAB[j++](1) = ptsB[assoc[i]](0)+ptsB[assoc[i]](1)*w;
      }
    }
    TOCK("Matching");

    tdp::SE3f T_ab;
    if (useHuber && assocAB.Area() > 5) {
      if (gui.verbose) std::cout << "huber" << std::endl;
      tdp::Huber3D3D<float> huber(pc, pcB, assocAB, huberDelta);
      huber.Compute(tdp::SE3f(), 1e-5, 100);
      T_ab =  huber.GetMinimum().Inverse();
      std::cout << T_ab << std::endl;
    }
    if (useRansac && assocAB.Area() > 5) {
      if (gui.verbose) std::cout << "ransac" << std::endl;
      tdp::P3P p3p;
      tdp::Ransac<tdp::Vector3fda> ransac(&p3p);
      size_t numInliers = 0;
      T_ab = ransac.Compute(pc, pcB, assocAB, ransacMaxIt,
          ransacThr, numInliers);

      std::vector<size_t> maxIt = {icpIter0, icpIter1, icpIter2};
      std::vector<float> errPerLvl;
      std::vector<float> countPerLvl;
      Eigen::Matrix<float,6,6> Sigma_ab = 1e-6*Eigen::Matrix<float,6,6>::Identity();
      if (runICP) {
        if (gui.verbose) std::cout << "icp" << std::endl;
        tdp::ICP::ComputeProjective<CameraT>(pcs_c, ns_c, pcs_m, ns_m,
            rig, rig.rgbStream2cam_, maxIt, icpLoopCloseAngleThr_deg, 
            icpLoopCloseDistThr,
            gui.verbose, T_ab, Sigma_ab, errPerLvl, countPerLvl);
        dH =  log(Sigma_ab.determinant());
        if (updatedEntropy) {
          dHkf = dH;
          updatedEntropy = false;
        }
      }

      float overlap = 0.;
      float rmse = 0.;
      tdp::Overlap(grey, greyB, pc, pcB, T_ab, cam, overlap, rmse);

      std::cout << "#inliers " << numInliers 
        << " %: " << numInliers /(float)assocAB.Area() 
        << " overlap " <<  overlap << " rmse " << rmse 
        << " dH/dHkf " << dH/dHkf << std::endl;

      logdH.Log(dH/dHkf, dEntropyThr);
      logOverlap.Log(overlap);
      logRmse.Log(rmse);
      logInliers.Log(numInliers/(float)assocAB.Area());

//      if (numInliers < ransacInlierPercThr*assocAB.Area()) {
//        T_ab = tdp::SE3f();
//      }
    }
    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    vbo.Upload(pcB.ptr_,pcB.SizeBytes(), 0);
    // render point cloud
    pangolin::glSetFrameOfReference(T_ab.matrix());
    pangolin::RenderVbo(vbo);
    pangolin::glUnsetFrameOfReference();

    for (size_t i=0; i < cosys.Area(); ++i) {
      pangolin::glDrawAxis(cosys[i].matrix(), 0.1f);
    }

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();

    plotOverlap.ScrollView(1,0);
    plotdH.ScrollView(1,0);
    plotRmse.ScrollView(1,0);
    plotInliers.ScrollView(1,0);
    // render normals image
    if (viewN2D.IsShown()) {
      // convert normals to RGB image
      tdp::Normals2Image(cuN, cuN2D);
      // copy normals image to CPU memory
      n2D.CopyFrom(cuN2D,cudaMemcpyDeviceToHost);
      viewN2D.SetImage(n2D);
    }
    if (viewGrey.IsShown()) {
      viewGrey.SetImage(grey);
      viewGrey.Activate();
      for (size_t i=0; i<ptsA.Area(); ++i) {
        glColor3f(1,0,0);
        pangolin::glDrawCross(ptsA[i](0), ptsA[i](1), 3);
        glColor3f(0,1,0);
        pangolin::glDrawLine(ptsA[i](0), ptsA[i](1), 
          ptsA[i](0)+cos(orientations[i])*10,
          ptsA[i](1)+sin(orientations[i])*10);
      } 
    }
    if (viewAssoc.IsShown()) {
      tdp::Image<uint8_t> greyAssocA = greyAssoc.GetRoi(0,0,wOrig, hOrig);
      tdp::Image<uint8_t> greyAssocB = greyAssoc.GetRoi(wOrig,0,wOrig, hOrig);
      greyAssocA.CopyFrom(grey, cudaMemcpyHostToHost);
      greyAssocB.CopyFrom(greyB, cudaMemcpyHostToHost);
      viewAssoc.SetImage(greyAssoc);
      viewAssoc.Activate();
      glColor3f(1,0,0);
      for (size_t i=0; i<ptsA.Area(); ++i) {
        glColor3f(1,0,0);
        pangolin::glDrawCross(ptsA[i](0), ptsA[i](1), 3);
        glColor3f(0,1,0);
        pangolin::glDrawLine(ptsA[i](0), ptsA[i](1), 
          ptsA[i](0)+cos(orientations[i])*10, ptsA[i](1)+sin(orientations[i])*10);
      }
      for (size_t i=0; i<ptsB.Area(); ++i) {
        glColor3f(1,0,0);
        pangolin::glDrawCross(ptsB[i](0)+wOrig, ptsB[i](1), 3);
        glColor3f(0,1,0);
        pangolin::glDrawLine(ptsB[i](0)+wOrig, ptsB[i](1), 
          ptsB[i](0)+wOrig+cos(orientationsB[i])*10, 
          ptsB[i](1)+sin(orientationsB[i])*10);
      }
      glColor3f(0,1,1);
      for (size_t i=0; i<ptsA.Area(); ++i) {
        if (assoc[i] >= 0)
          pangolin::glDrawLine(ptsA[i](0), ptsA[i](1), 
              ptsB[assoc[i]](0)+wOrig, ptsB[assoc[i]](1));
      }
    }

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


