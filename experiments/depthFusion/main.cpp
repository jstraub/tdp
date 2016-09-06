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
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/gldraw.h>

#include <Eigen/Dense>
#include <tdp/managed_image.h>

#include <tdp/convolutionSeparable.h>
#include <tdp/depth.h>
#include <tdp/normals.h>
#include <tdp/pc.h>
#include <tdp/quickView.h>
#include <tdp/volume.h>
#include <tdp/managed_volume.h>
#include <tdp/image.h>
#include <tdp/manifold/SE3.h>
#include <tdp/icp.h>
#include <tdp/tsdf.h>
#include <tdp/pyramid.h>
#include <tdp/managed_pyramid.h>
#include <tdp/nvidia/helper_cuda.h>

#include <tdp/Stopwatch.h>

#include "gui.hpp"

void VideoViewer(const std::string& input_uri, const std::string& output_uri)
{
  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return;
  }

  GUI gui(1200,800,video);
  size_t w = video.Streams()[gui.iRGB].Width();
  size_t h = video.Streams()[gui.iRGB].Height();
  size_t wc = w+w%64; // for convolution
  size_t hc = h+h%64;
  float f = 550;
  float uc = (w-1.)/2.;
  float vc = (h-1.)/2.;

  size_t dTSDF = 128;
  size_t wTSDF = 512;
  size_t hTSDF = 512;
  
  pangolin::Var<bool> dispNormalsPyrEst("ui.disp normal est", false, true);

  tdp::Camera<float> camView(Eigen::Vector4f(220,220,319.5,239.5)); 
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<Eigen::Matrix<uint8_t,3,1>> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<Eigen::Vector3f> n2Df(wc,hc);
  tdp::ManagedHostImage<Eigen::Vector3f> n(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedHostVolume<float> W(wTSDF, hTSDF, dTSDF);
  tdp::ManagedHostVolume<float> TSDF(wTSDF, hTSDF, dTSDF);
  W.Fill(0.);
  TSDF.Fill(-1.01);
  tdp::ManagedDeviceVolume<float> cuW(wTSDF, hTSDF, dTSDF);
  tdp::ManagedDeviceVolume<float> cuTSDF(wTSDF, hTSDF, dTSDF);

  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
  tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);

  tdp::ManagedHostImage<float> dEst(wc, hc);
  tdp::ManagedDeviceImage<float> cuDEst(wc, hc);
  dEst.Fill(0.);
  tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
  tdp::ManagedDeviceImage<float> cuDView(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcView(wc, hc);

  tdp::SE3<float> T_rd(Eigen::Matrix4f::Identity());
  tdp::Camera<float> camR(Eigen::Vector4f(f,f,uc,vc)); 
  tdp::Camera<float> camD(Eigen::Vector4f(f,f,uc,vc)); 

  // ICP stuff
  tdp::ManagedHostPyramid<float,3> dPyr(wc,hc);
  tdp::ManagedHostPyramid<float,3> dPyrEst(wc,hc);
  tdp::ManagedDevicePyramid<float,3> cuDPyr(wc,hc);
  tdp::ManagedDevicePyramid<float,3> cuDPyrEst(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);
  tdp::Matrix3fda R_mc = tdp::Matrix3fda::Identity();
  tdp::Vector3fda t_mc = tdp::Vector3fda::Zero();
  tdp::ManagedDeviceImage<float> cuICPassoc_m(wc, hc);
  tdp::ManagedDeviceImage<float> cuICPassoc_c(wc, hc);
  tdp::ManagedHostImage<float> ICPassoc_m(wc, hc);
  tdp::ManagedHostImage<float> ICPassoc_c(wc, hc);

  pangolin::GlBufferCudaPtr cuPcbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  tdp::ManagedHostImage<float> tsdfDEst(wc, hc);
  tdp::ManagedHostImage<float> tsdfSlice(wTSDF, hTSDF);
  tdp::QuickView viewTsdfDEst(wc,hc);
  tdp::QuickView viewTsdfSliveView(wTSDF,hTSDF);
  gui.container().AddDisplay(viewTsdfDEst);
  gui.container().AddDisplay(viewTsdfSliveView);

  tdp::ManagedHostImage<float> dispDepthPyr(dPyr.Width(0)+dPyr.Width(1), hc);
  tdp::QuickView viewDepthPyr(dispDepthPyr.w_,dispDepthPyr.h_);
  gui.container().AddDisplay(viewDepthPyr);
  
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuDispNormalsPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuDispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedHostImage<tdp::Vector3bda> dispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);

  tdp::QuickView viewNormalsPyr(dispNormals2dPyr.w_,dispNormals2dPyr.h_);
  gui.container().AddDisplay(viewNormalsPyr);

  tdp::QuickView viewICPassocM(wc,hc);
  gui.container().AddDisplay(viewICPassocM);
  tdp::QuickView viewICPassocC(wc,hc);
  gui.container().AddDisplay(viewICPassocC);

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<bool>  dispDepthPyrEst("ui.disp d pyr est", false,true);
  pangolin::Var<float> tsdfDmin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> tsdfDmax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool> fuseTSDF("ui.fuse TSDF",true,true);

  pangolin::Var<float> tsdfMu("ui.mu",0.5,0.,1.);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",dTSDF/2,0,dTSDF-1);
  pangolin::Var<float> grid0x("ui.grid0 x",-3.0,-2,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-3.0,-2,0);
  pangolin::Var<float> grid0z("ui.grid0 z",0.,0.,1);
  pangolin::Var<float> gridEx("ui.gridE x",3.0,2,0);
  pangolin::Var<float> gridEy("ui.gridE y",3.0,2,0);
  pangolin::Var<float> gridEz("ui.gridE z",3.5,2.,3);

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",25,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.15,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",2,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",0,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",0,0,10);

  pangolin::Var<float> offsettx("ui.tx",0.,-0.1,0.1);
  pangolin::Var<float> offsetty("ui.ty",0.,-0.1,0.1);
  pangolin::Var<float> offsettz("ui.tz",0.,-0.1,0.1);

  size_t numFused = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;

    tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
    tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
    tdp::Vector3fda dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);
    tdp::Vector3fda offsett(offsettx,offsetty,offsettz);

    T_rd.matrix().topRightCorner(3,1) += offsett;

    if (gui.verbose) std::cout << "ray trace" << std::endl;
    TICK("Ray Trace TSDF");
    RayTraceTSDF(cuTSDF, cuDEst, T_rd, camD, grid0, dGrid, tsdfMu); 
    TOCK("Ray Trace TSDF");

    if (gui.verbose) std::cout << "setup pyramids" << std::endl;
    TICK("Setup Pyramids");
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    ConvertDepthGpu(cuDraw, cuD, depthSensorScale, tsdfDmin, tsdfDmax);
    // construct pyramid  
    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
        cudaMemcpyDeviceToDevice);
    tdp::ConstructPyramidFromImage<float,3>(cuDEst, cuDPyrEst,
        cudaMemcpyDeviceToDevice);
    tdp::Depth2PCsGpu(cuDPyrEst,camD,pcs_m);
    tdp::Depth2PCsGpu(cuDPyr,camD,pcs_c);
    tdp::Depth2Normals(cuDPyrEst,camD,ns_m);
    tdp::Depth2Normals(cuDPyr,camD,ns_c);
    TOCK("Setup Pyramids");

    if (runICP && numFused > 30) {
      if (gui.verbose) std::cout << "icp" << std::endl;
      TICK("ICP");
      //T_rd.matrix() = Eigen::Matrix4f::Identity();
      tdp::SE3f dT;
      std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
      tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, dT,
          camD, maxIt, icpAngleThr_deg, icpDistThr); 
      T_rd.matrix() = dT.matrix()*T_rd.matrix();
      //std::cout << "T_mc" << std::endl << T_rd.matrix3x4() << std::endl;
      TOCK("ICP");
    }
    ICPassoc_c.Fill(NAN);
    ICPassoc_m.Fill(NAN);
    cuICPassoc_c.CopyFrom(ICPassoc_c,cudaMemcpyHostToDevice);
    cuICPassoc_m.CopyFrom(ICPassoc_m,cudaMemcpyHostToDevice);
    tdp::ICPVisualizeAssoc(pcs_m.GetImage(0), ns_m.GetImage(0),
        pcs_c.GetImage(0), ns_c.GetImage(0), T_rd,
          camD, icpAngleThr_deg, icpDistThr, cuICPassoc_m,
          cuICPassoc_c);

    if (pangolin::Pushed(resetTSDF)) {
      T_rd.matrix() = Eigen::Matrix4f::Identity();
      W.Fill(0.);
      TSDF.Fill(-1.01);
      dEst.Fill(0.);
      cuDEst.CopyFrom(dEst, cudaMemcpyHostToDevice);
      tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
      tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);
      numFused = 0;
      offsettx = 0.;
      offsetty = 0.;
      offsettz = 0.;
    }
    if (pangolin::Pushed(resetICP)) {
      T_rd.matrix() = Eigen::Matrix4f::Identity();
      offsettx = 0.;
      offsetty = 0.;
      offsettz = 0.;
    }

    if (fuseTSDF || numFused <= 30) {
      if (gui.verbose) std::cout << "add to tsdf" << std::endl;
      TICK("Add To TSDF");
      AddToTSDF(cuTSDF, cuW, cuD, T_rd, camD, grid0, dGrid, tsdfMu); 
      numFused ++;
      TOCK("Add To TSDF");
    }

    if (gui.verbose) std::cout << "draw 3D" << std::endl;
    TICK("Draw 3D");

    // Render point cloud from viewpoint of origin
    tdp::SE3f T_mv;
    RayTraceTSDF(cuTSDF, cuDView, T_mv, camView, grid0, dGrid, tsdfMu); 
    tdp::Depth2PCGpu(cuDView,camView,cuPcView);

    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // render model first
    pangolin::glDrawAxis(0.1f);
    {
      pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
      cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
      cudaMemcpy(*cuPcbufp, cuPcView.ptr_, cuPcView.SizeBytes(),
          cudaMemcpyDeviceToDevice);
    }
    glColor3f(0,0,1);
    pangolin::RenderVbo(cuPcbuf);
    pangolin::glSetFrameOfReference(T_rd.matrix());
    {
      pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
      cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
      tdp::Image<tdp::Vector3fda> pc0 = pcs_m.GetImage(0);
      cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
          cudaMemcpyDeviceToDevice);
    }
    glColor3f(0,1,0);
    pangolin::RenderVbo(cuPcbuf);
    // render current camera second in the propper frame of
    // reference
    pangolin::glDrawAxis(0.1f);
    {
      pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
      cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
      tdp::Image<tdp::Vector3fda> pc0 = pcs_c.GetImage(0);
      cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
          cudaMemcpyDeviceToDevice);
    }
    glColor3f(1,0,0);
    pangolin::RenderVbo(cuPcbuf);
    pangolin::glUnsetFrameOfReference();
    TOCK("Draw 3D");

    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);
    gui.ShowFrames();

    tsdfDEst.CopyFrom(cuDEst,cudaMemcpyDeviceToHost);
    viewTsdfDEst.SetImage(tsdfDEst);

    tdp::Image<float> cuTsdfSlice =
      cuTSDF.GetImage(std::min((int)cuTSDF.d_-1,tsdfSliceD.Get()));
    tsdfSlice.CopyFrom(cuTsdfSlice,cudaMemcpyDeviceToHost);
    viewTsdfSliveView.SetImage(tsdfSlice);

    if (dispDepthPyrEst) {
      tdp::PyramidToImage<float,3>(cuDPyrEst,dispDepthPyr,cudaMemcpyDeviceToHost);
    } else {
      tdp::PyramidToImage<float,3>(cuDPyr,dispDepthPyr,cudaMemcpyDeviceToHost);
    }
    viewDepthPyr.SetImage(dispDepthPyr);

    if (dispNormalsPyrEst) {
      tdp::PyramidToImage<tdp::Vector3fda,3>(ns_m,cuDispNormalsPyr,cudaMemcpyDeviceToDevice);
    } else {
      tdp::PyramidToImage<tdp::Vector3fda,3>(ns_c,cuDispNormalsPyr,cudaMemcpyDeviceToDevice);
    }
    tdp::Normals2Image(cuDispNormalsPyr, cuDispNormals2dPyr);
    dispNormals2dPyr.CopyFrom(cuDispNormals2dPyr,cudaMemcpyDeviceToHost);
    viewNormalsPyr.SetImage(dispNormals2dPyr);

    ICPassoc_m.CopyFrom(cuICPassoc_m,cudaMemcpyDeviceToHost);
    ICPassoc_c.CopyFrom(cuICPassoc_c,cudaMemcpyDeviceToHost);
    viewICPassocM.SetImage(ICPassoc_m);
    viewICPassocC.SetImage(ICPassoc_c);
    TOCK("Draw 2D");

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
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
