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
#include <tdp/nn/ann.h>
#include <tdp/manifold/S.h>

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>
#include "laplace_beltrami.h"


tdp::Vector3fda getMean(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  assert(pc.rows() == 1);
  tdp::Vector3fda mean;
  mean << 0,0,0;

  for (size_t i=0; i<nnIds.rows(); ++i){
      mean +=  pc(nnIds(i),0);
  }
  mean /= (float)nnIds.rows();
  return mean;
}

tdp::Matrix3fda getCovariance(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  // get covariance of the point cloud assuming no nan and pc of (nrows,1) size.
  assert (pc.rows() == 1);
  tdp::Matrix3fda cov;
  cov.setZero(3,3);

  tdp::Vector3fda mean = getMean(pc, nnIds);
  for(size_t i=0; i<nnIds.rows(); ++i){
    cov += (pc(nnIds(i),0)-mean)*(pc(nnIds(i),0)-mean).transpose();
  }
  cov /= (float)nnIds.rows();
  return cov;
}

tdp::ManagedHostImage<tdp::Vector3fda> GetSimplePc(){
    tdp::ManagedHostImage<tdp::Vector3fda> pc(10,1);
    for (size_t i=0; i<10; ++i){
        tdp::Vector3fda pt;
        pt << i,i,i;
        pc(i,0) = pt;
    }
    return pc;
}

void GetSphericalPc(tdp::Image<tdp::Vector3fda>& pc)
{
    for (size_t i=0; i<pc.w_; ++i) {
       pc[i] = tdp::S3f::Random().vector();
    }
}

void test1(){
    //TEST OF getMean and getCovariance
    tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();
    Eigen::VectorXi nnIds(2);
    nnIds<< 0,1;//,2;//,3,4,5,6,7,8,9;
    tdp::Vector3fda mean = getMean(pc, nnIds);
    tdp::Matrix3fda cov = getCovariance(pc,nnIds);
    std::cout << "mean: \n" << mean << std::endl << std::endl;
    std::cout << "cov: \n" << cov << std::endl << std::endl;
}

void getSortIds(const std::vector<auto>& vec, std::vector<int>& sortIds){
    int hi(0), lo(0), mid;
    for (int i=0; i<vec.size(); ++i){
        if (vec[i] < vec[lo]){
            lo = i;
        }else if (vec[i] > vec[hi]){
            hi = i;
        }
    }

    for (int i=0; i<vec.size();++i){
        if (i!=hi&&i!=lo){
            mid=i;
        }
    }
    sortIds.push_back(hi);
    sortIds.push_back(mid);
    sortIds.push_back(lo);
}

void test_getSortIds(){
    std::vector<int> v = {1,5,3};
    std::vector<int> ids;
    getSortIds(v,ids);
    for (int i =0; i<ids.size(); ++i){
        std::cout << ids[i] << ": "<< v[ids[i]] << std::endl;
    }
}

int main( int argc, char* argv[] ){
  // load pc and normal from the input paths
  tdp::ManagedHostImage<tdp::Vector3fda> pc(10/*1000*/,1);
  tdp::ManagedHostImage<tdp::Vector3fda> ns(1000,1);

  if (argc > 1) {
      const std::string input = std::string(argv[1]);
      std::cout << "input " << input << std::endl;
      tdp::LoadPointCloud(input, pc, ns);
  } else {
      //GetSphericalPc(pc);
      //todo: delete
      for (size_t i=0; i<10; ++i){
          tdp::Vector3fda pt;
          pt << i,i,i;
          pc(i,0) = pt;
      }
  }

  int n = 5;
  float eps = 1e-4;

  tdp::ANN ann;
  Eigen::VectorXi nnIds(n);
  Eigen::VectorXf dists(n);
  ann.ComputeKDtree(pc);
  tdp::Vector3fda query;
  tdp::Matrix3fda cov;
  Eigen::SelfAdjointEigenSolver<tdp::Matrix3fda> es;


  for (size_t i=0; i<pc.Area(); ++i){
      query = pc(i,0);
      ann.Search(query, n, eps, nnIds, dists);
      cov = getCovariance(pc,nnIds);
      es.compute(cov);

      std::vector<float> evalues;//todo: reinitialize
      std::vector<int> axesIds;//todo: reinitialize and move out of for loop
      for (size_t i=0; i<cov.rows(); ++i){
          float eval = std::real(es.eigenvalues().col(0)[i]);
          evalues.push_back( (eval<1e-6? 0: eval));
      }
      getSortIds(evalues,axesIds);

      Eigen::Matrix3f localBasis;
      for (size_t i=0; i<3; ++i){
          localBasis.col(i) = es.eigenvectors().col(axesIds[i]);
      }

      //change of basis
      for (size_t i=0; i<nnIds.size(); ++i){
        tdp::Vector3fda newPt = localBasis*pc(nnIds[i],0);
        std::cout << "newPt: \n" << newPt << std::endl;
      }


      /*
      //debug
      for (size_t i=0; i<evalues.size(); ++i){
        std::cout << sortIds[i] << ": "<< evalues[sortIds[i]] << std::endl;
        std::cout << "evec: \n" << es.eigenvectors().col(sortIds[i]) << std::endl;
      }

      std::cout << "eigenvalues: \n" << es.eigenvalues() << std::endl;
      std::cout << "eigenvectors: \n" << es.eigenvectors() << std::endl << std::endl;
      */
  }

/*
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
  pangolin::GlBuffer vbo, vboM;
  vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);

  // Add variables to pangolin GUI
  pangolin::Var<bool> runSkinning("ui.run skinning", false, false);
  pangolin::Var<int> knn("ui.knn", 10,1,100);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (pangolin::Pushed(runSkinning)) {
        //  processing of PC for skinning

      std::cout << "Running skinning..." << std::endl;

      // put the mean points to GLBuffer vboM
      // vboM.Reinitialise(pangolin::GlArrayBuffer, nMeans, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
      // vboM.Upload(&means[0], sizeof(float) * nMeans * 3, 0);

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
      pangolin::RenderVbo(vbo);
    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
*/
  std::cout << "good morning!" << std::endl;
  return 0;
}
