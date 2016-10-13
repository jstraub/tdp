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

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gui/gui.hpp>

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>
#include "skinning.h"

//void SkinningViewer(const std::string& input_uri)
//{
//  tdp::GUI gui(1200,800,video);

//  size_t w = video.Streams()[gui.iD[0]].Width();
//  size_t h = video.Streams()[gui.iD[0]].Height();
//  // width and height need to be multiple of 64 for convolution
//  // algorithm to compute normals.
//  w += w%64;
//  h += h%64;

//  // Define Camera Render Object (for view / scene browsing)
//  pangolin::OpenGlRenderState s_cam(
//      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
//      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
//      );
//  // Add named OpenGL viewport to window and provide 3D Handler
//  pangolin::View& d_cam = pangolin::CreateDisplay()
//    .SetHandler(new pangolin::Handler3D(s_cam));
//  gui.container().AddDisplay(d_cam);
//  // add a simple image viewer
//  tdp::QuickView viewN2D(w,h);
//  gui.container().AddDisplay(viewN2D);

//  // camera model for computing point cloud and normals
//  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5));

//  pangolin::GlBuffer vbo(pangolin::GlArrayBuffer,w*h,GL_FLOAT,3);
//  pangolin::GlBuffer cbo(pangolin::GlArrayBuffer,w*h,GL_UNSIGNED_BYTE,3);

//  // Add some variables to GUI
//  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
//  pangolin::Var<float> dMin("ui.d min",0.10,0.0,0.1);
//  pangolin::Var<float> dMax("ui.d max",4.,0.1,4.);
//  pangolin::Var<bool> savePC("ui.save current PC",false,false);
//}


void test_getMean(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
    tdp::Vector3fda mean = getMean(pc);
    std::cout << "mean: " << mean << std::endl;
}

void test_mean(){
  tdp::ManagedHostImage<tdp::Vector3fda> pc(10,10);
  for (int i=0; i<10;++i){
    tdp::Vector3fda p1,p2;
    p1 << i,2*i,0;
    //p2 << i,0,2*i;
    pc(i,0) = p1;
    //pc(i,1) = p2;
  }
  std::cout << "mean: " << getMean(pc) << std::endl;
  std::cout << "cov: \n" << getCovariance(pc) << std::endl;
}
void test_with_pc(const std::string inputA, const std::string inputB){
  // load pc and normal from the input paths
  tdp::ManagedHostImage<tdp::Vector3fda> pcA, pcB;
  tdp::ManagedHostImage<tdp::Vector3fda> nsA, nsB;
  tdp::LoadPointCloud(inputA, pcA, nsA);
  tdp::LoadPointCloud(inputB, pcB, nsB);

  tdp::Vector3fda meanA, meanB;
  tdp::Matrix3fda covA, covB;
  meanA = getMean(pcA); covA = getCovariance(pcA);
  meanB = getMean(pcB); covB = getCovariance(pcB);

  std::cout << "meanA: \n" << meanA << ",\nmeanB: " << meanB << std::endl;
  std::cout << "covA: \n " << covA << ", \ncovB:\n" << covB << std::endl;
}

tdp::Vector3fda getMean(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
  tdp::Vector3fda mean;
  mean << 0,0,0;
  int count = 0;
  for (int i=0; i<pc.w_; ++i){
    for (int j=0; j<pc.h_; ++j){
      if (not std::isnan(pc(i,j)[0]*pc(i,j)[1]*pc(i,j)[2])){ //check for nan
        mean +=  pc(i,j);
        count += 1;
      }
    }
  }
  return mean/count;
}

tdp::Matrix3fda getCovariance(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
  // get covariance of the point cloud
  tdp::Matrix3fda cov;
  cov.setZero(3,3);

  tdp::Vector3fda mean = getMean(pc);
  int count = 0;
  for(int x=0; x<pc.w_; ++x){
    for (int y=0; y<pc.h_; ++y){
      if (not std::isnan(pc(x,y)[0]*pc(x,y)[1]*pc(x,y)[2])){ //check for nan 
         cov += (pc(x,y)-mean)*(pc(x,y)-mean).transpose();
         count += 1;
        //std::cout << "x,y, and cov: " << x << ", " << y << ", " << cov << std::endl;
      }
    }
  }
  cov /= (float)(count);
  std::cout << "total number: " << (pc.w_*pc.h_) << std::endl;
  std::cout << "count : " << count << std::endl;
  return cov;
}

tdp::ManagedHostImage<tdp::Vector3fda> getSimplePc(){
  // PC for test
  tdp::ManagedHostImage<tdp::Vector3fda> pc(10,1);
    for (int i=0; i<10; i++){
            tdp::Vector3fda p(i,i,i);
            pc(i,0) = p;
    }
    std::cout << "test mean: \n" << getMean(pc) << std::endl;
    std::cout << "test cov: \n " << getCovariance(pc) << std::endl;
    return pc;
}
////todo: call getMeanAndSpreadOfBVoxel with correct p1 and p2
std::vector<tdp::Vector3fda> getMeanAndSpread(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
    tdp::Vector3fda mean = getMean(pc);
    //Eigen::MatrixXf mean = (Eigen::MatrixXf)getMean(pc);

    tdp::Matrix3fda cov = getCovariance(pc);
    std::cout << "mean: " << mean << std::endl;
    std::cout << "cov: " << cov << std::endl;

    Eigen::EigenSolver<tdp::Matrix3fda> es(cov);
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
    Eigen::VectorXcf spread = es.eigenvectors().col(maxIdx);
    tdp::Vector3fda spread_real, spread_imag;
    spread_real = (tdp::Vector3fda)spread.real();
    spread_imag = (tdp::Vector3fda)spread.imag();

    std::vector<tdp::Vector3fda> spec;
    spec[0] = mean;
    spec[1] = spread_real;
    spec[2] = spread_imag;
    return spec;
}
void test_getMeanAndSpread(){
  tdp::ManagedHostImage<tdp::Vector3fda> pc = getSimplePc();
  std::vector<tdp::Vector3fda> meanSpread;
  meanSpread = getMeanAndSpread(pc);
  tdp::Vector3fda mean, spread_real;
  mean = meanSpread[0]; spread_real = meanSpread[1];

  std::cout << "test mean: \n" << mean << std::endl;
  std::cout << "test spread: \n" << spread_real<< std::endl;
}
inline bool inBVoxel(const tdp::Vector3fda& p, const tdp::Vector3fda& topLeft, const tdp::Vector3fda& btmRight){
    return topLeft[0]<=p[0] && p[0]<btmRight[0] && topLeft[1]<=p[1] && p[1]<btmRight[1] && topLeft[2]<=p[2] && p[2]<btmRight[2];
}

std::vector<tdp::Vector3fda> meanAndSpreadOfBVoxel(const tdp::ManagedHostImage<tdp::Vector3fda>& pc, const tdp::Vector3fda& p1, const tdp::Vector3fda& p2){
    tdp::Vector3fda topLeft, btmRight;
    tdp::Vector3fda mean; //todo: check this?
    mean << 0,0,0;
    // Find the correct bounding voxel's coordinates
    for (int i=0; i<3; ++i){
        topLeft[i] = std::min(p1[i],p2[i]);
        btmRight[i] = std::max(p1[i],p2[i]);
    }
    //Calculate mean
    //overhead
    //Todo: implement BVoxelId (image of the same size as pc where each entry is BVoxel id)
    int count = 0;
    std::vector<tdp::Vector3fda> points;
    for (int i=0; i<pc.w_; ++i){
        for (int j=0; j<pc.h_; ++j){
            if (inBVoxel(pc(i,j), topLeft, btmRight)){
                mean += pc(i,j);
                points.push_back(pc(i,j));
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
    Eigen::EigenSolver<tdp::Matrix3fda> es(cov);
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
    Eigen::VectorXcf spread = es.eigenvectors().col(maxIdx);
    //Eigen::VectorXf spread_real, spread_imag;
    tdp::Vector3fda spread_real, spread_imag;
    spread_real = (tdp::Vector3fda)spread.real(); 
    spread_imag = (tdp::Vector3fda)spread.imag();


    std::vector<tdp::Vector3fda> spec;
    spec[0] = mean;
    spec[1] = spread_real;
    spec[2] = spread_imag;

    return spec;
}

std::vector<tdp::Vector3fda> getMeans(const tdp::ManagedHostImage<tdp::Vector3fda>& pc, int nsteps){
  ///nsteps in the positive/negative direction. totalsteps is 2*nsteps.

  // find the mean and eigenvector 
  std::vector<tdp::Vector3fda> meanAndSpread = getMeanAndSpread(pc); 
  tdp::Vector3fda mean, spread_real, spread_imag, stepVec; 
  mean = meanAndSpread[0];
  spread_real = meanAndSpread[1];
  spread_imag = meanAndSpread[2];
  float spread_size, step_size;
  spread_size = spread_real.norm(); //todo: get the magnitude of complex //todo: eigen's norm?
  step_size = spread_size/nsteps;
  stepVec = step_size*(spread_real/spread_size);//todo: what to do with spread_imag?

  tdp::Vector3fda start1 = mean;
  tdp::Vector3fda end1 = mean+stepVec;
  tdp::Vector3fda start2 = mean;
  tdp::Vector3fda end2 = mean-stepVec;

  std::vector<tdp::Vector3fda> means;
  means.push_back(mean);
  for (int i=1; i<=nsteps; ++i){
    std::vector<tdp::Vector3fda> meanAndCov_pos = meanAndSpreadOfBVoxel(pc, start1, end1);
    std::vector<tdp::Vector3fda> meanAndCov_neg = meanAndSpreadOfBVoxel(pc, start2, end2);
    means.push_back(meanAndCov_pos[0]);
    means.push_back(meanAndCov_neg[0]);

    start1 = end1;
    end1 += stepVec;
    start2 = end2;
    end2 -= stepVec;
  }
return means;
}

int main( int argc, char* argv[] ){
  test_getMeanAndSpread();
  if (argc < 2){
      std::cout << "Must input two plyfile paths!" << std::endl;
      return -1;
  }
  //todo: send the points to draw with opengl
  const std::string inputA = std::string(argv[1]);
  const std::string inputB = std::string(argv[2]);
  test_with_pc(inputA, inputB);
  std::cout << "inputA, B: " << inputA << ", " << inputB << std::endl;
  std::cout << "argc: " << argc << std::endl;
  const std::string option = (argc > 3) ? std::string(argv[3]) : "";

  bool runOnce = false;
  if (!option.compare("-1")) {
    runOnce = true;
  }

  // Create OpenGL window - guess sensible dimensions
  int menue_w = 180;
  pangolin::CreateWindowAndBind( "GuiBase", 1200+menue_w, 800);
  // current frame in memory buffer and displaying.
  pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menue_w));
  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // setup container
  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
    .SetBounds(0., 1.0, pangolin::Attach::Pix(menue_w), 1.0);
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewPc);
  pangolin::View& viewN = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewN);

  // load pc and normal from the input paths
  tdp::ManagedHostImage<tdp::Vector3fda> pcA, pcB;
  tdp::ManagedHostImage<tdp::Vector3fda> nsA, nsB;
  tdp::LoadPointCloud(inputA, pcA, nsA);
  tdp::LoadPointCloud(inputB, pcB, nsB);

//  std::cout << "pcA area:  " << pcA.Area() << std::endl;
//  std::cout << "pcB area:  " << pcB.Area() << std::endl;

  // use those OpenGL buffers
  pangolin::GlBuffer vboA, vboB, vboM;
  vboA.Reinitialise(pangolin::GlArrayBuffer, pcA.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vboA.Upload(pcA.ptr_, pcA.SizeBytes(), 0);
  vboB.Reinitialise(pangolin::GlArrayBuffer, pcB.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vboB.Upload(pcB.ptr_, pcB.SizeBytes(), 0);

  std::cout << "Vboxa: " << vboA.num_elements << std::endl;
  std::cout << "Vboxb: " << vboB.num_elements << std::endl;
  // Add variables to pangolin GUI
  pangolin::Var<bool> runSkinning("ui.run skinning", false, false);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    if (runOnce) break;
    if (pangolin::Pushed(runSkinning)) {
      //  processing of PC for skinning
      glColor3f(1.0f, 1.0f, 1.0f);

      std::cout << "Running skinning..." << std::endl;
      int nSteps = 10;
      std::vector<tdp::Vector3fda> means = getMeans(pcA,nSteps);
      size_t nMeans= means.size();
      std::cout << "number of means (should be 2*nsteps + 1):" << nMeans << std::endl;

      // put the mean points to GLBuffer vboM
      vboM.Reinitialise(pangolin::GlArrayBuffer, nMeans, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
      vboM.Upload(&means[0], sizeof(float) * nMeans * 3, 0);

    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);
    if (viewPc.IsShown()) {
      viewPc.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(10.);
      glColor3f(1.0f, 0.0f, 1.0f);
      pangolin::RenderVbo(vboM);

      glPointSize(1.);
      // draw the first arm pc
      glColor3f(1.0f, 0.0f, 0.0f);
      pangolin::RenderVbo(vboA);

      // draw the second arm pc
      glColor3f(0., 1., 0.);
      pangolin::RenderVbo(vboB);
    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }

  std::cout << "good morning!" << std::endl;
  return 0;
}
