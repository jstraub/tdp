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

  size_t dTSDF = 64;
  size_t wTSDF = wc;
  size_t hTSDF = hc;

  gui.tsdfSliceD.Meta().range[1] = dTSDF-1;
  gui.tsdfSliceD = dTSDF/2;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisY)
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
  tdp::ManagedHostImage<float> dEst(wTSDF, hTSDF);
  W.Fill(0.);
  TSDF.Fill(-gui.tsdfMu);
  dEst.Fill(0.);
  tdp::ManagedDeviceVolume<float> cuW(wTSDF, hTSDF, dTSDF);
  tdp::ManagedDeviceVolume<float> cuTSDF(wTSDF, hTSDF, dTSDF);
  tdp::ManagedDeviceImage<float> cuDEst(wTSDF, hTSDF);

  tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
  tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);

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

  pangolin::GlBufferCudaPtr cuPcbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  tdp::ManagedHostImage<float> debugA(wTSDF, hTSDF);
  tdp::ManagedHostImage<float> debugB(wTSDF, hTSDF);
  tdp::QuickView viewDebugA(wTSDF,hTSDF);
  tdp::QuickView viewDebugB(wTSDF,hTSDF);
  gui.container().AddDisplay(viewDebugA);
  gui.container().AddDisplay(viewDebugB);
  size_t numFused = 0;
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;

    pangolin::basetime t0 = pangolin::TimeNow();
    CopyImage(dRaw, cuDraw, cudaMemcpyHostToDevice);
    ConvertDepth(cuDraw, cuD, 1e-4, 0.1, 4.);
    // construct pyramid  
    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
        cudaMemcpyDeviceToDevice);
    tdp::ConstructPyramidFromImage<float,3>(cuDEst, cuDPyrEst,
        cudaMemcpyDeviceToDevice);
    tdp::Depth2PCs(cuDPyrEst,camD,pcs_m);
    tdp::Depth2PCs(cuDPyr,camD,pcs_c);
    tdp::Depth2Normals(cuDPyrEst,camD,ns_m);
    tdp::Depth2Normals(cuDPyr,camD,ns_c);

    if (gui.runICP && numFused > 30) {
      //T_rd.matrix() = Eigen::Matrix4f::Identity();
      std::vector<size_t> maxIt{gui.icpIter0,gui.icpIter1,gui.icpIter2};
      tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, T_rd,
          camD, maxIt, gui.icpAngleThr_deg, gui.icpDistThr); 
      //std::cout << "T_mc" << std::endl << T_rd.matrix3x4() << std::endl;
    }
    pangolin::basetime tDepth = pangolin::TimeNow();

    if (pangolin::Pushed(gui.resetTSDF)) {
      T_rd.matrix() = Eigen::Matrix4f::Identity();
      W.Fill(0.);
      TSDF.Fill(-1.01);
      dEst.Fill(0.);
      tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
      tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
      tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);
      numFused = 0;
    }

    gui.tsdfRho0 = 1./gui.tsdfDmax;
    gui.tsdfDRho = (1./gui.tsdfDmin - gui.tsdfRho0)/float(dTSDF-1);

    AddToTSDF(cuTSDF, cuW, cuD, T_rd, camR, camD, gui.tsdfRho0,
        gui.tsdfDRho, gui.tsdfMu); 
    numFused ++;

    checkCudaErrors(cudaDeviceSynchronize());
    pangolin::basetime tAddTSDF = pangolin::TimeNow();

    RayTraceTSDF(cuTSDF, cuDEst, T_rd, camR, camD, gui.tsdfRho0,
        gui.tsdfDRho, gui.tsdfMu); 

    checkCudaErrors(cudaDeviceSynchronize());
    pangolin::basetime tRayTrace = pangolin::TimeNow();

    std::cout << "splits: " << pangolin::TimeDiff_s(t0,tDepth) << "\t"
      << pangolin::TimeDiff_s(tDepth,tAddTSDF) << "\t"
      << pangolin::TimeDiff_s(tAddTSDF,tRayTrace) << "\t"
      << pangolin::TimeDiff_s(t0,tRayTrace) << "\t"<< std::endl;

    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // render model first
    Eigen::Matrix4f T_w = Eigen::Matrix4f::Identity();
    pangolin::glSetFrameOfReference(T_w);
    pangolin::glDrawAxis(0.1f);
    {
      pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
      cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
      tdp::Image<tdp::Vector3fda> pc0 = pcs_m.GetImage(0);
      cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
          cudaMemcpyDeviceToDevice);
    }
    glColor3f(0,1,0);
    pangolin::RenderVbo(cuPcbuf);
    pangolin::glUnsetFrameOfReference();
    // render current camera second in the propper frame of
    // reference
    pangolin::glSetFrameOfReference(T_rd.matrix());
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

    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);
    gui.ShowFrames();

    CopyImage(cuDEst, debugA, cudaMemcpyDeviceToHost);
    //if (gui.tsdfSliceD.GuiChanged()) {
    tdp::Image<float> sliceTSDF(cuTSDF.w_, cuTSDF.h_,
        cuTSDF.ImagePtr(std::min((int)cuTSDF.d_-1,gui.tsdfSliceD.Get())));
    CopyImage(sliceTSDF, debugB, cudaMemcpyDeviceToHost);
    //}
    viewDebugA.SetImage(debugA);
    viewDebugB.SetImage(debugB);

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
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
