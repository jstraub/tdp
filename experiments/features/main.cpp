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

void VideoViewer(const std::string& input_uri, const std::string& output_uri)
{

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return;
  }

  tdp::GUI gui(1200,800,video);

  size_t w = video.Streams()[gui.iD[0]].Width();
  size_t h = video.Streams()[gui.iD[0]].Height();
  size_t wOrig = video.Streams()[gui.iD[0]].Width();
  size_t hOrig = video.Streams()[gui.iD[0]].Height();
  // width and height need to be multiple of 64 for convolution
  // algorithm to compute normals.
  w += w%64;
  h += h%64;
  size_t wc = w;
  size_t hc = h;

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
  pangolin::Var<float> ransacMaxIt("ui.max it",3000,1,1000);
  pangolin::Var<float> ransacThr("ui.thr",0.03,0.01,1.0);
  pangolin::Var<float> ransacInlierPercThr("ui.inlier % thr",0.15,0.1,1.0);

  pangolin::Var<int> fastB("ui.FAST b",30,0,100);
  pangolin::Var<float> harrisThr("ui.harris thr",0.1,0.001,2.0);
  pangolin::Var<float> kappaHarris("ui.kappa harris",0.08,0.04,0.15);
  pangolin::Var<int> briefMatchThr("ui.BRIEF match",65,0,100);
  pangolin::Var<bool> newKf("ui.new KF", false, false);

  tdp::ManagedHostImage<tdp::Brief> descsA;
  tdp::ManagedHostImage<tdp::Brief> descsB;

  tdp::ManagedHostImage<tdp::Vector2ida> ptsA;
  tdp::ManagedHostImage<float> orientations;
  tdp::ManagedHostImage<float> orientationsB;
  tdp::ManagedHostImage<tdp::Vector2ida> ptsB;

  tdp::ManagedHostImage<tdp::SE3f> cosys;

  tdp::ManagedHostImage<int32_t> assoc;

  tdp::ManagedHostImage<tdp::Vector2ida> assocAB;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (gui.frame == 1 || pangolin::Pushed(newKf)) {
      descsB.Reinitialise(descsA.w_, descsA.h_);
      descsB.CopyFrom(descsA, cudaMemcpyHostToHost);
      ptsB.Reinitialise(ptsA.w_, 1);
      ptsB.CopyFrom(ptsA, cudaMemcpyHostToHost);
      greyB.CopyFrom(grey, cudaMemcpyHostToHost);
      pcB.CopyFrom(pc, cudaMemcpyHostToHost);
      orientationsB.Reinitialise(ptsA.w_,1);
      orientationsB.CopyFrom(orientations, cudaMemcpyHostToHost);
    }

    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    // get rgb image
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGreyOrig,1.);
    tdp::Image<uint8_t> cuGrey0 = cuPyrGrey.GetImage(0);
    tdp::Blur5(cuGreyOrig,cuGrey0, 10.);
    tdp::CompletePyramid(cuPyrGrey, cudaMemcpyDeviceToDevice);

    // get depth image
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    // copy raw image to gpu
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    // convet depth image from uint16_t to float [m]
    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
    // compute point cloud (on CPU)
    tdp::Depth2PC(d,cam,pc);
    // compute normals
    tdp::Depth2Normals(cuD, cam, cuN);


    int fastLvl = 0;
//    tdp::Blur9(cuGrey,cuGreyChar, 10.);
    grey.CopyFrom(cuPyrGrey.GetImage(fastLvl), cudaMemcpyDeviceToHost);
    pyrGrey.CopyFrom(cuPyrGrey, cudaMemcpyDeviceToHost);
    TICK("Detection");
    tdp::DetectOFast(grey, fastB, kappaHarris, harrisThr, 16, ptsA, orientations);
    TOCK("Detection");

    TICK("Extraction");
//    tdp::ExtractBrief(grey, ptsA, orientations, descsA);
    tdp::ExtractBrief(pyrGrey, ptsA, orientations, gui.frame, fastLvl, descsA);
    TOCK("Extraction");

    tdp::Gradient3D(cuGrey, cuD, cuN, cam, 0.001f, cuGreydu,
        cuGreydv, cuGrad3D);
    grad3D.CopyFrom(cuGrad3D, cudaMemcpyDeviceToHost);
    n.CopyFrom(cuN, cudaMemcpyDeviceToHost);

    cosys.Reinitialise(ptsA.w_,1);
    for (size_t i=0; i<ptsA.Area(); ++i) {
      const tdp::Vector3fda& pci = pc(ptsA[i](0),ptsA[i](1));
      const tdp::Vector3fda& ni = n(ptsA[i](0),ptsA[i](1));
      tdp::Vector3fda gradi = grad3D(ptsA[i](0),ptsA[i](1)).normalized();
      tdp::SO3f R = tdp::SO3f::FromOrthogonalVectors(ni, gradi);
      cosys[i] = tdp::SE3f(R, pci);
    }

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
      tdp::Huber3D3D<float> huber(pc, pcB, assocAB, huberDelta);
      huber.Compute(tdp::SE3f(), 1e-5, 100);
      T_ab =  huber.GetMinimum().Inverse();
      std::cout << T_ab << std::endl;
    }
    if (useRansac && assocAB.Area() > 5) {
      tdp::P3P p3p;
      tdp::Ransac<tdp::Vector3fda> ransac(&p3p);
      size_t numInliers = 0;
      T_ab = ransac.Compute(pc, pcB, assocAB, ransacMaxIt,
          ransacThr, numInliers);
      std::cout << "#inliers " << numInliers 
        << " %: " << numInliers /(float)assocAB.Area() << std::endl;
      if (numInliers < ransacInlierPercThr*assocAB.Area()) {
        T_ab = tdp::SE3f();
      }
    }

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


int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";

  if( argc > 1 ) {
    const std::string input_uri = std::string(argv[1]);
    const std::string output_uri = (argc > 2) ? std::string(argv[2]) : dflt_output_uri;
    try{
      VideoViewer(input_uri, output_uri);
    } catch (pangolin::VideoException e) {
      std::cout << e.what() << std::endl;
    }
  }else{
    const std::string input_uris[] = {
      "dc1394:[fps=30,dma=10,size=640x480,iso=400]//0",
      "convert:[fmt=RGB24]//v4l:///dev/video0",
      "convert:[fmt=RGB24]//v4l:///dev/video1",
      "openni:[img1=rgb]//",
      "test:[size=160x120,n=1,fmt=RGB24]//"
        ""
    };

    std::cout << "Usage  : VideoViewer [video-uri]" << std::endl << std::endl;
    std::cout << "Where video-uri describes a stream or file resource, e.g." << std::endl;
    std::cout << "\tfile:[realtime=1]///home/user/video/movie.pvn" << std::endl;
    std::cout << "\tfile:///home/user/video/movie.avi" << std::endl;
    std::cout << "\tfiles:///home/user/seqiemce/foo%03d.jpeg" << std::endl;
    std::cout << "\tdc1394:[fmt=RGB24,size=640x480,fps=30,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tdc1394:[fmt=FORMAT7_1,size=640x480,pos=2+2,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tv4l:///dev/video0" << std::endl;
    std::cout << "\tconvert:[fmt=RGB24]//v4l:///dev/video0" << std::endl;
    std::cout << "\tmjpeg://http://127.0.0.1/?action=stream" << std::endl;
    std::cout << "\topenni:[img1=rgb]//" << std::endl;
    std::cout << std::endl;

    // Try to open some video device
    for(int i=0; !input_uris[i].empty(); ++i )
    {
      try{
        pango_print_info("Trying: %s\n", input_uris[i].c_str());
        VideoViewer(input_uris[i], dflt_output_uri);
        return 0;
      }catch(pangolin::VideoException) { }
    }
  }

  return 0;
}
