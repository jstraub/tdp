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
#include <tdp/preproc/lab.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/normals.h>
#include <tdp/preproc/lab.h>

#include <tdp/gui/gui.hpp>
#include <tdp/clustering/dpmeans.hpp>


class Kmeans{
public:
  //todo: threshold as input (or default)
  Kmeans(int k) : k_(k) {};
  ~Kmeans() {};

  void Compute(const tdp::Image<tdp::Vector3fda>& x,
               size_t maxIt, float threshold);

  int k_;
  tdp::eigen_vector<tdp::Vector3fda> centers_;
  std::vector<int> labels_;

private:
  void UpdateLabels(
    const tdp::Image<tdp::Vector3fda>& x
    );
  void UpdateCenters(
    const tdp::Image<tdp::Vector3fda>& x,
    int option
    );
};

void Kmeans::Compute(const tdp::Image<tdp::Vector3fda>& x,
                     size_t maxIt, float threshold){
  Initialize(centers_);
  tdp::eigen_vector<tdp::Vector3fda> newCenters;
  int nIter = 0;
  while( newCenters.size() <= 0 || 
         ( (newCenter - centers).squaredNorm()>threshold && nIter<maxIter ) ){
    Kmeans::UpdateLabels(x, centers_);
    Kmeans::ComputeCenters(x, option);
  }

}

void Kmeans::UpdateLabels(const tdp::Image<tdp::Vector3fda>& x){

}

void Kmeans::UpdateCenters(const tdp::Image<tdp::Vector3fda>& x,
                           int option){

}


int main( int argc, char* argv[] )
{
  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

  // read input data

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
  tdp::QuickView viewLab(w,h);
  gui.container().AddDisplay(viewLab);
  tdp::QuickView viewZ(w,h);
  gui.container().AddDisplay(viewZ);

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> lab(w, h);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,
  pangolin::Var<bool> recomputeMeans("ui.recomp means", true, true);


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
    tdp::Rgb2LabCpu(rgb, lab);
    // get depth image
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    cudaMemset(cuDraw.ptr_, 0, cuDraw.SizeBytes());
    // copy raw image to gpu
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    // convet depth image from uint16_t to float [m]
    tdp::ConvertDepthGpu(cuDraw, cuD, depthSensorScale, dMin, dMax);
    d.CopyFrom(cuD, cudaMemcpyDeviceToHost);
    if (recomputeMeans) {
      std::pair<double,double> minMax = d.MinMax();
      for (size_t i=0; i<abd.Area(); ++i) {
        if (d[i]==d[i]) {
          abd[i](0) = lab[i](1)/128.;
          abd[i](1) = lab[i](2)/128.;
          abd[i](2) = alpha*(d[i]-minMax.first)/(minMax.second-minMax.first);
        } else {
          abd[i] << NAN,NAN,NAN;
        }
      }
      vbo.Upload(abd.ptr_,abd.SizeBytes(), 0);
      cuAbd.CopyFrom(abd, cudaMemcpyHostToDevice);
      dpMeans.lambda_ = lambda;
      dpMeans.Compute(abd, cuAbd, cuZ, maxIt, minNchangePerc);
//      std::pair<double,double> minMax = abd.MinMax();
//      std::cout << minMax.first << " " << minMax.second << std::endl;
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    if (d_cam.IsShown()) {
      d_cam.Activate(s_cam);
      // draw the axis
      pangolin::glDrawAxis(0.1);
      // render point cloud
      pangolin::RenderVbo(vbo);
    }
    glDisable(GL_DEPTH_TEST);

    // Draw 2D stuff
    // SHowFrames renders the raw input streams (in our case RGB and D)
    gui.ShowFrames();

    if (viewLab.IsShown()) {
      tdp::Rgb2LabCpu(rgb, lab8);
      viewLab.SetImage(lab8);
    }
    if (viewZ.IsShown()) {
      z.CopyFrom(cuZ, cudaMemcpyDeviceToHost);
      for (size_t i=0; i<z.Area(); ++i)
        if (z[i] > dpMeans.K_)
          z[i] = dpMeans.K_+1;
      viewZ.SetImage(z);
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
  
  return 0;
}


