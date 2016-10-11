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
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/gui/gui.hpp>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>

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

  size_t w = video.Streams()[0].Width();
  size_t h = video.Streams()[0].Height();
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

  // camera model for computing point cloud and normals
  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 
  
  // host image: image in CPU memory
  tdp::ManagedHostImage<float> d(w, h);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(w, h);
  tdp::ManagedHostImage<tdp::Vector3bda> n2D(w, h);

  // device image: image in GPU memory
  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(w, h);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuN2D(w, h);

  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

  // Add some variables to GUI
  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);

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

tdp::Vector3fda getMean(pc){
  tdp::Vector3fda mean(0,0,0);
  for (int i=0; i<pc.w; ++i){
    for (int j=0; y<pc.h; ++j){
        mean +=  pc(i,j);
    }
  }
  return mean/(pc.w*pc.h);
}

tdp::Matrix3fda getCovariance(tdp::ManagedHostImage<tdp::Vector3fda> pc){
  // get covariance of the point cloud
  tdp::Matrix3fda cov;
  cov.setZero(3,3);

  tdp::Vector3fda mean = getMean(pc);
  for(int x=0; x<pc.w_; ++x){
    for (int y=0; y<pc.h_; ++y){
        cov += (pc(x,y)-mean)*(pc(x,y)-mean).transpose();
        std::cout << "x,y, and cov: " << x << ", " << y << ", " << cov << std::endl;
    }
  }
  cov /= (float)(pc.w_*pc.h_);
  std::cout << "final: " << cov << std::endl;
  return cov;
}

tdp::ManagedHostImage<tdp::Vector3fda> getSimplePc(){
  // PC for test  
  tdp::ManagedHostImage<tdp::Vector3fda> pc(10,10);
    for (int i=0; i<10; i++){
            tdp::Vector3fda p(i,0,0);
            pc(i) = p;
        }
    return pc;
}

//todo: call getMeanAndSpreadOfBVoxel with correct p1 and p2
std::vector<tdp::Vector3fda> getMeanAndSpread(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
    tdp::Vector3fda mean = getMean(pc);
    tdp::Matrix3fda cov = getCovariance(pc);
    std::cout << "mean: " << mu << std::endl;
    std::cout << "cov: " << cov << std::endl;

    Eigen::EigenSolver<MatrixXd> es(cov);
    std::cout << "eigenvalues: " << es.eigenvalues() << std::endl;
    std::cout << "eigenvectors: " << es.eigenvectors() << std::endl << std::endl;

    std::complex<float> maxEval(-1,0);
    int maxIdx(-1);
    for (int i=0; i< cov.rows();++i ){
        if (abs(maxEval) < abs(es.eigenvalues().col(0)[i])){
            maxEval = es.eigenvalues().col(0)[i];
            maxIdx = i;
        }
    }
    tdp::Vector3fda spread = es.eigenvector().col(maxIdx);
    std::vector spec = {mu, spread};
    return spec;
}
inline bool inBVoxel(const tdp::Vector3fda& p, const tdp::Vector3fda& topLeft, const tdp::Vector3fda btmRight){
    return topLeft[0]<=p[0] && p[0]<btmRight[0] && topLeft[1]<=p[1] && p[1]<btmRight[1] && topLeft[2]<=p[2] && p[2]<btmRight[2];
}

vector<tdp::Vector3fda> meanAndSpreadOfBVoxel(const tdp::ManagedDeviceImage<tdp::Vector3fda>& pc, const tdp::Vector3fda& p1, const tdp::Vector3fda& p2){
    tdp::Vector3fda topLeft, btmRight;
    tdp::Vector3fda mean(0,0,0); //todo: check this?
    // Find the correct bounding voxel's coordinates
    for (int i=0; i<3; ++i){
        topLeft[i] = std::min(p1[i],p2[i]);
        btmRight[i] = std::max(p1[i],p2[i]);
    }
    //Calculate mean
    //overhead
    //Todo: implement BVoxelId (image of the same size as pc where each entry is BVoxel id)
    count = 0; 
    vector<tdp::Vect3fda> points;
    for (int i=0; i<pc.w_; ++i){
        for (int j=0; j<pc.h_; ++j){
            if inBVoxel(p, topLeft, btmRight){
                mean += p;
                points.puch_back(p);
                count += 1; 
	    }
        }
    }
    mean /= count; 
    // calculate covariance
    tdp::Matrix3fda cov;
    cov.setZero(3,3);
    for (int i=0; i<count; ++i){
      cov += (points[i]-mean)*(points[i]-mean).transpose();
    }
    cov /= count;
    std::cout << "final: " << cov << std::endl;
    
    // eigenvector
    Eigen::EigenSolver<MatrixXd> es(cov);
    std::cout << "eigenvalues: " << es.eigenvalues() << std::endl;
    std::cout << "eigenvectors: " << es.eigenvectors() << std::endl << std::endl;

    std::complex<float> maxEval(-1,0);
    int maxIdx(-1);
    for (int i=0; i< cov.rows();++i ){
        if (abs(maxEval) < abs(es.eigenvalues().col(0)[i])){
            maxEval = es.eigenvalues().col(0)[i];
            maxIdx = i;
        }
    }
    tdp::Vector3fda spread = es.eigenvector().col(maxIdx);
    std::vector spec = {mean, spread};
    return spec;

}


vector<tdp::Vector3fda> getMeans(const tdp::ManagedDeviceImage<tdp::Vector3fda>& pc, const int nsteps){
  //nsteps in the positive/negative direction. totalsteps is 2*nsteps.

  // find the mean and eigenvector -> its size is eigenvalue
  vector<tdp::Vector3fda> meanAndSpread = getMeanAndSpread(pc);
  tdp::Vector3fda mean, spread, stepVec;
  mean = meanAndSPread[0];
  spread = meanAndSpread[1];
  tdp::Vector3fda direction = pc.spread_/abs(pc.spread);

  float spread_size, step_size;
  spread_size = norm(spread);
  step_size = spread_size/nsteps;
  stepVec = step_size*(spread/spread_size);

  tdp::Vector3fda start1 = mean;
  tdp::Vector3fda end1 = mean+stepVec;
  tdp::Vector3fda start2 = mean;
  tdp::Vector3fda end2 = mean-stepVec;

  vector<tdp::Vect3fda> means;
  for (int i=1; i<=nsteps; ++i){
      vector<tdp::Vector3fda> meanAndCov_pos = meanAndCovOfBVoxel(pc, start1, end1);
      vector<tdp::Vector3fda> meanAndCov_neg = meanAndCovOfBVoxel(pc, start2, end2);
      means.push_back(meanAndCov_pos[0], meanAndCov_neg[0]);
   
      start1 = end1;
      end1 += stepVec;
      start2 = end2;
      end2 -= stepVec;
  }
}

int main( int argc, char* argv[] )
{
  //todo: send the points (in 3d) to draw to opengl
  //
  return 0;
}
