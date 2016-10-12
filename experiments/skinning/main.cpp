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

#include <tdp/gui/gui.hpp>
#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>
#include "skinning.h"

//test
void test_getMean(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
    tdp::Vector3fda mean = getMean(pc);
    std::cout << "mean: " << mean << std::endl;
}

tdp::Vector3fda getMean(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
  tdp::Vector3fda mean;
  mean << 0,0,0;
  for (int i=0; i<pc.w_; ++i){
    for (int j=0; j<pc.h_; ++j){
        mean +=  pc(i,j);
    }
  }
  return mean/(pc.w_*pc.h_);
}

tdp::Matrix3fda getCovariance(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
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
            pc(i,0) = p;
        }
    return pc;
}

////todo: call getMeanAndSpreadOfBVoxel with correct p1 and p2
std::vector<Eigen::MatrixXf> getMeanAndSpread(const tdp::ManagedHostImage<tdp::Vector3fda>& pc){
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
    Eigen::VectorXf spread_real, spread_imag;
    spread_real = spread.real();
    spread_imag = spread.imag();

    std::vector<Eigen::MatrixXf> spec;
    spec[0] = (Eigen::MatrixXf)mean;
    spec[1] = (Eigen::MatrixXf)spread_real;
    spec[2] = (Eigen::MatrixXf)spread_imag;
    return spec;
}

//inline bool inBVoxel(const tdp::Vector3fda& p, const tdp::Vector3fda& topLeft, const tdp::Vector3fda& btmRight){
//    return topLeft[0]<=p[0] && p[0]<btmRight[0] && topLeft[1]<=p[1] && p[1]<btmRight[1] && topLeft[2]<=p[2] && p[2]<btmRight[2];
//}

//std::vector<tdp::Vector3fda> meanAndSpreadOfBVoxel(const tdp::ManagedDeviceImage<tdp::Vector3fda>& pc, const tdp::Vector3fda& p1, const tdp::Vector3fda& p2){
//    tdp::Vector3fda topLeft, btmRight;
//    tdp::Vector3fda mean(0,0,0); //todo: check this?
//    // Find the correct bounding voxel's coordinates
//    for (int i=0; i<3; ++i){
//        topLeft[i] = std::min(p1[i],p2[i]);
//        btmRight[i] = std::max(p1[i],p2[i]);
//    }
//    //Calculate mean
//    //overhead
//    //Todo: implement BVoxelId (image of the same size as pc where each entry is BVoxel id)
//    count = 0;
//    vector<tdp::Vect3fda> points;
//    for (int i=0; i<pc.w_; ++i){
//        for (int j=0; j<pc.h_; ++j){
//            if (inBVoxel(pc(i,j), topLeft, btmRight)){
//                mean += pc(i,j);
//                points.puch_back(pc(i,j));
//                count += 1;
//            }
//        }
//    }
//    mean /= count;
//    // calculate covariance
//    tdp::Matrix3fda cov;
//    cov.setZero(3,3);
//    for (int i=0; i<count; ++i){
//      cov += (points[i]-mean)*(points[i]-mean).transpose();
//    }
//    cov /= count;
//    std::cout << "final: " << cov << std::endl;
    
//    // eigenvector
//    Eigen::EigenSolver<MatrixXd> es(cov);
//    std::cout << "eigenvalues: " << es.eigenvalues() << std::endl;
//    std::cout << "eigenvectors: " << es.eigenvectors() << std::endl << std::endl;

//    std::complex<float> maxEval(-1,0);
//    int maxIdx(-1);
//    for (int i=0; i< cov.rows();++i ){
//        if (abs(maxEval) < abs(es.eigenvalues().col(0)[i])){
//            maxEval = es.eigenvalues().col(0)[i];
//            maxIdx = i;
//        }
//    }
//    tdp::Vector3fda spread = es.eigenvector().col(maxIdx);
//    std::vector spec = {mean, spread};
//    return spec;
//}

//std::vector<tdp::Vector3fda> getMeans(const tdp::ManagedDeviceImage<tdp::Vector3fda>& pc, const int nsteps){
//  //nsteps in the positive/negative direction. totalsteps is 2*nsteps.

//  // find the mean and eigenvector -> its size is eigenvalue
//  vector<tdp::Vector3fda> meanAndSpread = getMeanAndSpread(pc);
//  tdp::Vector3fda mean, spread, stepVec;
//  mean = meanAndSPread[0];
//  spread = meanAndSpread[1];

//  float spread_size, step_size;
//  spread_size = norm(spread);
//  step_size = spread_size/nsteps;
//  stepVec = step_size*(spread/spread_size);

//  tdp::Vector3fda start1 = mean;
//  tdp::Vector3fda end1 = mean+stepVec;
//  tdp::Vector3fda start2 = mean;
//  tdp::Vector3fda end2 = mean-stepVec;

//  vector<tdp::Vect3fda> means;
//  for (int i=1; i<=nsteps; ++i){
//      vector<tdp::Vector3fda> meanAndCov_pos = meanAndCovOfBVoxel(pc, start1, end1);
//      vector<tdp::Vector3fda> meanAndCov_neg = meanAndCovOfBVoxel(pc, start2, end2);
//      means.push_back(meanAndCov_pos[0], meanAndCov_neg[0]);
   
//      start1 = end1;
//      end1 += stepVec;
//      start2 = end2;
//      end2 -= stepVec;
//  }
//}

int main( int argc, char* argv[] )
{
  //todo: send the points (in 3d) to draw to opengl
  //
  const std::string inputA = std::string(argv[1]);
  const std::string inputB = std::string(argv[2]);
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
  // use those OpenGL buffers

  tdp::ManagedHostImage<tdp::Vector3fda> pcA, pcB;
  tdp::ManagedHostImage<tdp::Vector3fda> nsA, nsB;
  tdp::LoadPointCloud(inputA, pcA, nsA);
  tdp::LoadPointCloud(inputB, pcB, nsB);

  return 0;
}
