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
#include <tdp/preproc/grad.h>
#include <tdp/preproc/grey.h>
#include <tdp/manifold/SL3.h>
#include <tdp/esm/esm.h>
#include <tdp/data/managed_pyramid.h>

#include <tdp/gui/gui.hpp>


int main( int argc, char* argv[] )
{
  std::string input_uri = std::string(argv[1]);
  std::string output_uri = (argc > 2) ? std::string(argv[2]) : "pango://video.pango";

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

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
  // Add named OpenGL viewport to window and provide 3D Handler pangolin::View& d_cam = pangolin::CreateDisplay()
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);
  // add a simple image viewer
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);

  tdp::QuickView viewGrey(3*w/2,h);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewGreyDu(3*w/2,h);
  gui.container().AddDisplay(viewGreyDu);
  tdp::QuickView viewGreyDv(3*w/2,h);
  gui.container().AddDisplay(viewGreyDv);
  tdp::QuickView viewGrey_m(3*w/2,h);
  gui.container().AddDisplay(viewGrey_m);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);
  tdp::ManagedHostImage<float> greyVis(3*w/2, h);
  tdp::ManagedHostImage<float> greydu(3*w/2, h);
  tdp::ManagedHostImage<float> greydv(3*w/2, h);
  tdp::ManagedHostImage<float> grey(w, h);
  tdp::ManagedHostImage<float> grey_m(w, h);
  tdp::ManagedHostImage<float> greyVis_m(3*w/2, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);

  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(w, h);
  tdp::ManagedDeviceImage<float> cuGrey(w, h);
  tdp::ManagedDeviceImage<float> cuGreydu(w, h);
  tdp::ManagedDeviceImage<float> cuGreydv(w, h);

  tdp::ManagedDevicePyramid<float,5> cuPyrGrey(w, h);
  tdp::ManagedDevicePyramid<float,5> cuPyrGreydv(w, h);
  tdp::ManagedDevicePyramid<float,5> cuPyrGreydu(w, h);

  tdp::ManagedDevicePyramid<float,5> cuPyrGrey_m(w, h);
  tdp::ManagedDevicePyramid<float,5> cuPyrGreydv_m(w, h);
  tdp::ManagedDevicePyramid<float,5> cuPyrGreydu_m(w, h);

  tdp::ManagedHostPyramid<float,5> pyrGrey(w, h);
  tdp::ManagedHostPyramid<float,5> pyrGreydv(w, h);
  tdp::ManagedHostPyramid<float,5> pyrGreydu(w, h);

  tdp::ManagedHostPyramid<float,5> pyrGrey_m(w, h);
  tdp::ManagedHostPyramid<float,5> pyrGreydv_m(w, h);
  tdp::ManagedHostPyramid<float,5> pyrGreydu_m(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> randomH("ui.random H",true,true);
  pangolin::Var<bool> identityH("ui.unit H",false,true);
  pangolin::Var<bool> estimateH("ui.estimate H",false,false);

  pangolin::Var<int>   esmIter0("ui.ESM iter lvl 0",2,0,10);
  pangolin::Var<int>   esmIter1("ui.ESM iter lvl 1",0,0,10);
  pangolin::Var<int>   esmIter2("ui.ESM iter lvl 2",0,0,10);
  pangolin::Var<int>   esmIter3("ui.ESM iter lvl 3",0,0,10);
  pangolin::Var<int>   esmIter4("ui.ESM iter lvl 4",0,0,10);

  tdp::Homography<float> H_rand(tdp::SL3<float>::Random());
  tdp::Homography<float> H_est;
  tdp::SL3<float> H;
  Eigen::Matrix3f Kinv = cam.GetKinv();
  Eigen::Matrix3f K = cam.GetK();
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    gui.NextFrames();

    // get rgb image
    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    cudaMemset(cuRgb.ptr_, 0, cuRgb.SizeBytes());
    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGrey);
    tdp::Gradient(cuGrey, cuGreydu, cuGreydv);

    tdp::ConstructPyramidFromImage(cuGrey, cuPyrGrey,
        cudaMemcpyDeviceToDevice);
    tdp::ConstructPyramidFromImage(cuGreydu, cuPyrGreydu,
        cudaMemcpyDeviceToDevice);
    tdp::ConstructPyramidFromImage(cuGreydv, cuPyrGreydv,
        cudaMemcpyDeviceToDevice);

    tdp::PyramidToImage(cuPyrGrey, greyVis, cudaMemcpyDeviceToHost);
    tdp::PyramidToImage(cuPyrGreydu, greydu, cudaMemcpyDeviceToHost);
    tdp::PyramidToImage(cuPyrGreydv, greydv, cudaMemcpyDeviceToHost);

    pyrGrey.CopyFrom(cuPyrGrey, cudaMemcpyDeviceToHost);
    pyrGreydu.CopyFrom(cuPyrGreydu, cudaMemcpyDeviceToHost);
    pyrGreydv.CopyFrom(cuPyrGreydv, cudaMemcpyDeviceToHost);
    
    if (randomH && !gui.paused()) {
      H_rand = tdp::Homography<float>::Random();
      if (identityH)
        H_rand = tdp::Homography<float>();
      std::cout << "true random homography" << std::endl 
        << H_rand.matrix() << std::endl;

      grey.CopyFrom(cuGrey,cudaMemcpyDeviceToHost);
      tdp::Transform(grey, H_rand, grey_m);
      tdp::ConstructPyramidFromImage(grey_m, cuPyrGrey_m,
          cudaMemcpyHostToDevice);
      tdp::Image<float> greydv_m0 = cuPyrGreydv_m.GetImage(0);
      tdp::Image<float> greydu_m0 = cuPyrGreydu_m.GetImage(0);
      tdp::Gradient(cuPyrGrey_m.GetImage(0), greydu_m0, greydv_m0);
      tdp::CompletePyramid(cuPyrGreydu_m, cudaMemcpyDeviceToDevice);
      tdp::CompletePyramid(cuPyrGreydv_m, cudaMemcpyDeviceToDevice);

      pyrGrey_m.CopyFrom(  cuPyrGrey_m,   cudaMemcpyDeviceToHost);
      pyrGreydu_m.CopyFrom(cuPyrGreydv_m, cudaMemcpyDeviceToHost);
      pyrGreydv_m.CopyFrom(cuPyrGreydu_m, cudaMemcpyDeviceToHost);
    }

    if (gui.frame > 1 && pangolin::Pushed(estimateH)) {
      tdp::SL3<float> G;
      std::vector<size_t> maxIt{esmIter0,esmIter1,esmIter2,esmIter3,esmIter4};
      tdp::ESM::EstimateHomography(
          pyrGrey_m, pyrGreydu_m, pyrGreydv_m,
          pyrGrey, pyrGreydu, pyrGreydv,
          maxIt,
          G);

      H_est = G;
      H = tdp::SL3<float>(Kinv * G.matrix() * K);
      
      Eigen::Vector3f m(0,0,1.);
      Eigen::Vector3f dt = - (H.matrix() - Eigen::Matrix3f::Identity())*m;
      Eigen::Vector3f omega = - tdp::SO3f::vee(H.matrix() - H.matrix().transpose());
      Eigen::Matrix<float,6,1> dx;
      dx << omega, dt;
      tdp::SE3f dT(tdp::SE3f::Exp_(dx));
      std::cout << dT << std::endl; 

      float x,y;
      int corners[10] = {0,0,640,0,640,480,0,480,0,0};
      for (size_t i=0; i<10; i+=2) {
        H_rand.Transform(corners[i],corners[i+1],x,y);
        std::cout << "True H corners: " << x << ", " << y << std::endl;
        H_est.Transform(corners[i],corners[i+1],x,y);
        std::cout << "Est H corners: " << x << ", " << y << std::endl;
      }


//      tdp::SE3f dT;
//      Eigen::Vector3f n;
//      tdp::Homography<float>(H.matrix()).ToPoseAndNormal(dT, n);
      gui.pause();
    }

    if (!gui.paused()) {
      tdp::PyramidToImage(pyrGrey_m, greyVis_m, cudaMemcpyHostToHost);
      pyrGrey_m.CopyFrom(pyrGrey, cudaMemcpyHostToHost);
      pyrGreydu_m.CopyFrom(pyrGreydv, cudaMemcpyHostToHost);
      pyrGreydv_m.CopyFrom(pyrGreydu, cudaMemcpyHostToHost);
    }

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
    // convert normals to RGB image
    tdp::Normals2Image(cuN, cuN2D);
    // copy normals image to CPU memory
    n2D.CopyFrom(cuN2D,cudaMemcpyDeviceToHost);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    pangolin::glDrawAxis(0.1);
    vbo.Upload(pc.ptr_,pc.SizeBytes(), 0);
    cbo.Upload(rgb.ptr_,rgb.SizeBytes(), 0);
    // render point cloud
    pangolin::RenderVboCbo(vbo,cbo,true);

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();
    // render normals image
    if (viewN2D.IsShown()) {
      viewN2D.SetImage(n2D);
    }
    if (viewGrey.IsShown()) {
      viewGrey.SetImage(greyVis);
      viewGrey.Activate();

      glColor3f(0.,1.,0.);
      glLineWidth(3);
      float x,y,xPrev,yPrev;
      int corners[10] = {0,0,640,0,640,480,0,480,0,0};
      for (size_t i=0; i<10; i+=2) {
        H_rand.Transform(corners[i],corners[i+1],x,y);
        if (i>2)
          pangolin::glDrawLine(xPrev,yPrev,x,y);
       xPrev = x; yPrev = y;
      }
      glColor3f(1.,0.,0.);
      glLineWidth(3);
      for (size_t i=0; i<10; i+=2) {
        H_est.Transform(corners[i],corners[i+1],x,y);
        if (i>2)
          pangolin::glDrawLine(xPrev,yPrev,x,y);
       xPrev = x; yPrev = y;
      }
    }
    if (viewGreyDu.IsShown()) {
      viewGreyDu.SetImage(greydu);
    }
    if (viewGreyDv.IsShown()) {
      viewGreyDv.SetImage(greydv);
    }
    if (viewGrey_m.IsShown()) {
      viewGrey_m.SetImage(greyVis_m);
//      viewGrey_m.SetImage(grey_m);
    }


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
  return 0;
}


