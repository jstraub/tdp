/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <algorithm>
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
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#include <tdp/preproc/grey.h>
#include <tdp/preproc/grad.h>
#include <tdp/preproc/adaptiveThreshold.h>
#endif

#include <tdp/gui/gui.hpp>

/// Suppress non-maxima by examining the norm values in plus minus
/// gradient direction.
void NonMaxSuppression(
    tdp::Image<uint8_t> Iedge,
    tdp::Image<float> Inorm,
    tdp::Image<float> Itheta
    ) {
  for (int x=1; x<(int)Iedge.w_-1; ++x) 
    for (int y=1; y<(int)Iedge.h_-1; ++y) {
      if (Iedge(x,y) > 128) {
        float norm = Inorm(x,y);
        float ang = Itheta(x,y);
        float xp = std::max(0.f,std::min((float)Iedge.w_-1,x+cosf(ang)));
        float yp = std::max(0.f,std::min((float)Iedge.h_-1,y+sinf(ang)));
        float xn = std::max(0.f,std::min((float)Iedge.w_-1,x-cosf(ang)));
        float yn = std::max(0.f,std::min((float)Iedge.h_-1,y-sinf(ang)));
        float normP = Inorm.GetBilinear(xp,yp);
        float normN = Inorm.GetBilinear(xn,yn);
        if (norm <= normP || norm <= normN) {
          Iedge(x,y) = 0;
        }
      }
    }
}

/// Find contours in an edge image (no edge I<128; edge I>=128)
uint16_t FindContours(
    tdp::Image<uint8_t> Iedge,
    tdp::Image<uint16_t> cId,
    uint16_t minLen
    )
{
  const int w = Iedge.w_;
  const int h = Iedge.h_;
  cId.Fill(0);
  int16_t id = 1;
  std::vector<uint16_t> cCounts;
  for (size_t i=0; i<Iedge.Area(); ++i) {
    if (Iedge[i] > 128) {
      cCounts.push_back(1);
      // found a starting point for a contour
      // lok at starting location twice to get both directions
      for (size_t j=0; j<2; ++j) {
        int x = i%w;  
        int y = i/w;  
        cId[i] = id;
        Iedge[i] = 0; 
        // follow contour 
        bool foundPath = true;
        while (foundPath) {
          int r = (std::min(w-1,x+1)) + y*w;
          if (Iedge[r] > 128) {
            cId[r] = id;
            Iedge[r] = 0; 
            x = r%w; y = r/w;
            cCounts[id-1] ++;
            continue;
          } 
          int rd = (std::min(w-1,x+1)) + std::min(h-1,y+1)*w;
          if (Iedge[rd] > 128) {
            cId[rd] = id;
            Iedge[rd] = 0; 
            x = rd%w; y = rd/w;
            cCounts[id-1] ++;
            continue;
          } 
          int d = x + std::min(h-1,y+1)*w;
          if (Iedge[d] > 128) {
            cId[d] = id;
            Iedge[d] = 0; 
            x = d%w; y = d/w;
            cCounts[id-1] ++;
            continue;
          } 
          int ld = (std::max(0,x-1)) + std::min(h-1,y+1)*w;
          if (Iedge[ld] > 128) {
            cId[ld] = id;
            Iedge[ld] = 0; 
            x = ld%w; y = ld/w;
            cCounts[id-1] ++;
            continue;
          } 
          int l = (std::max(0,x-1)) + y*w;
          if (Iedge[l] > 128) {
            cId[l] = id;
            Iedge[l] = 0; 
            x = l%w; y = l/w;
            cCounts[id-1] ++;
            continue;
          } 
          int lu = (std::max(0,x-1)) + std::max(0,y-1)*w;
          if (Iedge[lu] > 128) {
            cId[lu] = id;
            Iedge[lu] = 0; 
            x = lu%w; y = lu/w;
            cCounts[id-1] ++;
            continue;
          } 
          int u = x + std::max(0,y-1)*w;
          if (Iedge[u] > 128) {
            cId[u] = id;
            Iedge[u] = 0; 
            x = u%w; y = u/w;
            cCounts[id-1] ++;
            continue;
          } 
          int ru = (std::min(w-1,x+1)) + std::max(0,y-1)*w;
          if (Iedge[ru] > 128) {
            cId[ru] = id;
            Iedge[ru] = 0; 
            x = ru%w; y = ru/w;
            cCounts[id-1] ++;
            continue;
          } 
          foundPath = false;
//          if (cId[l] > 0 || cId[lu] > 0 || cId[u] > 0 || cId[ru] > 0 || 
//              cId[r] > 0 || cId[rd] > 0 || cId[d] > 0 || cId[ld] > 0) {
//            std::cout << "collided with other contour "
//              << id  << " "
//              <<  cCounts[id-1] << ": "
//              << cId[l] << "," 
//              << cId[lu] << "," 
//              << cId[u] << "," 
//              << cId[ru] << "," 
//              << cId[r] << "," 
//              << cId[rd] << "," 
//              << cId[d] << "," 
//              << cId[ld] << "," 
//              << std::endl;
//          }
        }
      }
      id ++;
    }
  }
  for (size_t i=0; i<cId.Area(); ++i) {
    if (cId[i] > 0) {
      if (cCounts[cId[i]-1] < minLen) {
        cId[i] = 0;
      }
    }
  }

  // count number of contours passing threshold
  id = 0;
  for (auto count: cCounts) 
    if (count >= minLen) 
      id++;
  return id;
}


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
  size_t wOrig = w;
  size_t hOrig = h;
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
  tdp::QuickView viewN2D(w,h);
  gui.container().AddDisplay(viewN2D);
  tdp::QuickView viewGrey(w,h);
  gui.container().AddDisplay(viewGrey);
  tdp::QuickView viewGreyDu(w,h);
  gui.container().AddDisplay(viewGreyDu);
  tdp::QuickView viewGreyDv(w,h);
  gui.container().AddDisplay(viewGreyDv);
  tdp::QuickView viewGreyDnorm(w,h);
  gui.container().AddDisplay(viewGreyDnorm);
  tdp::QuickView viewGreyDtheta(w,h);
  gui.container().AddDisplay(viewGreyDtheta);
  tdp::QuickView viewGreyDnormThr(w,h);
  gui.container().AddDisplay(viewGreyDnormThr);
  tdp::QuickView viewContourId(w,h);
  gui.container().AddDisplay(viewContourId);
  tdp::QuickView viewThr(w,h);
  gui.container().AddDisplay(viewThr);
  tdp::QuickView viewThrBinary(w,h);
  gui.container().AddDisplay(viewThrBinary);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);
  tdp::ManagedHostImage<float> thr(w, h);
  tdp::ManagedHostImage<float> grey(w, h);
  tdp::ManagedHostImage<uint8_t> thrBinary(w, h);
  tdp::ManagedHostImage<float> greydu(w, h);
  tdp::ManagedHostImage<float> greydv(w, h);

  tdp::ManagedHostImage<float> greydnorm(w, h);
  tdp::ManagedHostImage<float> greydtheta(w, h);
  tdp::ManagedHostImage<uint8_t> dNormThr(w, h);

  tdp::ManagedHostImage<uint16_t> cId(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<float> cuGrey(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuRgb(wOrig, hOrig);
  tdp::ManagedDeviceImage<float> cuThr(w, h);
  tdp::ManagedDeviceImage<uint8_t> cuThrBinary(w, h);
  tdp::ManagedDeviceImage<uint8_t> cuDnormThr(w, h);

  tdp::ManagedDeviceImage<float> cuGreydu(w, h);
  tdp::ManagedDeviceImage<float> cuGreydv(w, h);
  tdp::ManagedDeviceImage<float> cuGreydnorm(w, h);
  tdp::ManagedDeviceImage<float> cuGreydtheta(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);
  pangolin::Var<int> thrDiameter("ui.thr D",5,3,9);
  pangolin::Var<float> threshold("ui.thr",5.,-1.,10.);
  pangolin::Var<float> thresholdGradNorm("ui.thr grad norm",30.,10.,40.);
  pangolin::Var<int> minContourLen("ui.min len",20,1,100);
  pangolin::Var<int> numContours("ui.# contours",0,0,0);

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
    // get depth image
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
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

    cuRgb.CopyFrom(rgb,cudaMemcpyHostToDevice);
    tdp::Rgb2Grey(cuRgb,cuGrey);
    grey.CopyFrom(cuGrey,cudaMemcpyDeviceToHost);

    tdp::AdaptiveThreshold(cuGrey,cuThr,thrDiameter);
    thr.CopyFrom(cuThr,cudaMemcpyDeviceToHost);

    tdp::AdaptiveThreshold(cuGrey,cuThrBinary,thrDiameter, threshold);
    thrBinary.CopyFrom(cuThrBinary,cudaMemcpyDeviceToHost);

    Gradient(cuGrey, cuGreydu, cuGreydv);
    Gradient2AngleNorm(cuGreydu,cuGreydv,cuGreydtheta, cuGreydnorm);
    greydu.CopyFrom(cuGreydu, cudaMemcpyDeviceToHost);
    greydv.CopyFrom(cuGreydv, cudaMemcpyDeviceToHost);
    greydnorm.CopyFrom(cuGreydnorm, cudaMemcpyDeviceToHost);
    greydtheta.CopyFrom(cuGreydtheta, cudaMemcpyDeviceToHost);

    tdp::Threshold(cuGreydnorm,cuDnormThr, thresholdGradNorm);
    dNormThr.CopyFrom(cuDnormThr,cudaMemcpyDeviceToHost);

    tdp::Image<uint8_t> Iedge(w,h,dNormThr.ptr_);
    NonMaxSuppression(Iedge, greydnorm, greydtheta);
    numContours = FindContours(Iedge, cId, minContourLen);

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
    viewN2D.SetImage(n2D);
    viewThr.SetImage(thr);
    viewThrBinary.SetImage(thrBinary);
    viewGrey.SetImage(grey);
    viewGreyDu.SetImage(greydu);
    viewGreyDv.SetImage(greydv);
    viewGreyDnorm.SetImage(greydnorm);
    viewGreyDtheta.SetImage(greydtheta);
    viewGreyDnormThr.SetImage(dNormThr);
    viewContourId.SetImage(cId);

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
