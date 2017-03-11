#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <iterator>

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
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <Eigen/Core>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/gl_draw.h>

#include <tdp/gui/gui.hpp>
#include <tdp/gui/quickView.h>

#include <tdp/nn/ann.h>
#include <tdp/manifold/S.h>
#include <tdp/manifold/SE3.h>
#include <tdp/data/managed_image.h>

#include <tdp/utils/status.h>
#include <tdp/utils/timer.hpp>
#include <tdp/eigen/std_vector.h>
#include <tdp/eigen/dense_io.h>
#include <tdp/eigen/sparse_io.h>

#include <unistd.h> //check if cached files exist

#include <tdp/laplace_beltrami/laplace_beltrami.h>
/************Declarations***************************************
 ***************************************************************/


/************TODO***********************************************/
/***************************************************************/
//1. shape deformation
//2. try function transfer + get error on deformed sphere


int main(int argc, char* argv[]){

  // Add variables to pangolin GUI
  pangolin::Var<bool> showFMap("ui.show fMap", true, false);
  pangolin::Var<bool> showFTransfer(" ui. show fTransfer", true, true);

  pangolin::Var<int> nSamples("ui. num samples from mesh pc", 1000, 100, 3000);
  pangolin::Var<int> shapeOpt("ui. shape option", 1, 0, 3); //2:bunny

  //--second shape point cloud
    pangolin::Var<float> sFactor("ui. sFactor", 1, 1, 10); // spherical deformation using phi angle
  pangolin::Var<float> max_phi("ui. max phi", 1e-6, 1e-6, M_PI); // spherical deformation using phi angle
  pangolin::Var<float> noiseStd("ui. noiseStd", 0, 0.00001, 0.0001); //zero mean Gaussian noise added to shape S


  //-- variables for KNN
  pangolin::Var<int> knn("ui.knn",30,1,100);//(int)nSamples
  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);
  pangolin::Var<float> alpha("ui. alpha", 0.01, 0.001, 0.3); //variance of rbf kernel for laplacian


  //--Correspondence Matrix C estimation
  pangolin::Var<float> alpha2("ui. alpha2", 0.0001, 0.0001, 0.01); //variance of rbf kernel for defining function on manifold
  pangolin::Var<int> nEv("ui.num Ev",50,30,100); //min=1, max=pc_s.Area()
  pangolin::Var<int> nPW("ui.num PointWise train",nSamples/*std::min(20*numEv, pc_s.Area())*/,nEv,nSamples);



  //*****************Declare variables*********************************/
 std::string option("rbf");

 tdp::ManagedHostImage<tdp::Vector3fda> pc_all, pc_s_spherical, pc_t_spherical, pc_s, pc_t; 
 std::string fpath_b, fpath_m;
 tdp::ANN ann_s, ann_t;

  Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area()),//Laplacian of manifold S
                             L_t(pc_t.Area(), pc_t.Area());//Laplacian of manifold T
  Eigen::VectorXf S_evals((int)nEv),//evalues of Laplacian of S. Increasing order.
                  T_evals((int)nEv);//evalues of Laplacian of T. Increasing order.

  Eigen::MatrixXf S_wl((int)nSamples,(int)nEv),
                  T_wl((int)nSamples,(int)nEv),
                  F((int)nPW, (int)nEv),
                  G((int)nPW, (int)nEv),
                  C,//((int)nEv, (int)nEv);
                  D;//DiffMap;(pc_s.Area(), pc_s.Area())
  std::vector<int> dIndices; //Indices to deformed points
  std::vector<int> pIndices; //For functional correpsondence pairs
  /*********************************************************************/

  // if (argc<3){
  //   std::cout << "Give paths to bunny and manekine ply files. Terminated." << std::endl;
  //   return -1;
  // } else{
  //   fpath_b = argv[1];
  //   fpath_m = argv[2];
  // }
  // std::cout << "\n\nshape opt: " << shapeOpt << std::endl;
  shapeOpt = 1;

        // Get point clouds
  switch (shapeOpt){
    case 0:
    tdp::GetSimplePc(pc_s, nSamples);
    pc_t.ResizeCopyFrom(pc_s);
    std::cout << "Shape: linear---" << std::endl;
    break;

    case 1:
//            tdp::GetSphericalPc(pc_s, nSamples);
    tdp::GetPointsOnSphere(pc_s_spherical, (int)nSamples);
    tdp::toCartisean(pc_s_spherical, pc_s);
    dIndices = tdp::Deform(pc_s_spherical, pc_t_spherical, (float)max_phi);
    tdp::toCartisean(pc_t_spherical, pc_t);

    std::cout << "Shape: sphere---" << std::endl;
    break;

    case 2:
    tdp::LoadPointCloudFromMesh(fpath_b, pc_all);
    tdp::GetSamples(pc_all, pc_s, nSamples);
    pc_t.ResizeCopyFrom(pc_s);
    std::cout << "Shape: bunny---" << std::endl;
    break;

    case 3:
    tdp::ManagedHostImage<tdp::Vector3fda> pc_all;
    tdp::LoadPointCloudFromMesh(fpath_m, pc_all);
    tdp::GetSamples(pc_all, pc_s, nSamples);
    pc_t.ResizeCopyFrom(pc_s);
    std::cout << "Shape: manekine---" << std::endl;
  }
  //debug
  std::cout << "deformed indicies: " << std::endl;
  for (auto i = dIndices.begin(); i != dIndices.end(); ++i ){
    std::cout << *i << " ";
  }
  std::cout << std::endl;

        // Modify second shape
  //TODO: Test on deformed shapes
        tdp::scale(pc_s, sFactor, pc_t); //scale
        //tdp::addGaussianNoise(pc_t, (float)noiseStd, pc_t);

        //Get indices to be used (later) for correspondence between functions
  std::random_device rd;
  std::mt19937 g(rd());

  for (int i = 0; i<(int)nSamples; ++i){
    pIndices.push_back(i);
  }

        //shuffle correpsondence indices
  std::shuffle(pIndices.begin(), pIndices.end(), g);

        // build kd tree
  ann_s.ComputeKDtree(pc_s);
  ann_t.ComputeKDtree(pc_t);
        //***************Get Laplacians***************************//
  L_s = tdp::getLaplacian(pc_s, ann_s, knn, eps, alpha);
  L_t = tdp::getLaplacian(pc_t, ann_t, knn, eps, alpha);

  tdp::Timer t0;
  std::cout << "Calculating Bases&evals---" << std::endl;
  tdp::decomposeLaplacian(L_s, nEv, S_evals, S_wl); //todo: check if size initialization is included
  tdp::decomposeLaplacian(L_t, nEv, T_evals, T_wl);
  t0.toctic("Laplacian decomposition");

  tdp::gramSchmidt(S_wl);
  tdp::gramSchmidt(T_wl);
  t0.toctic("gramSchmidt");

 // --construct F(data matrix) and G based on the correspondences
 for (int i=0; i< (int)nPW; ++i){
    Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area()), f_l, g_l;

    tdp::f_landmark(pc_s, pIndices[i], alpha2, option, f_w); //points in suffled order
    tdp::f_landmark(pc_t, pIndices[i], alpha2, option, g_w);

    f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
    g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);

    F.row(i) = f_l.transpose();
    G.row(i) = g_l.transpose();
  }

// solve least-square
  C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);
  t0.toctic("fmap");

  std::cout << "-----------\n"
  << "C(10x10) \n"
  << C.block(0,0,10,10)
  << std::endl;

  std::cout << "----------\n"
  << "Diagnoals\n"
  << C.diagonal().transpose()
  << std::endl;

// calculate DiffMap
  D = C.transpose()*C;
  t0.toctic("shape diff map");
  std::cout << "D: " << std::endl;
  std::cout << D.diagonal().transpose()<< std::endl;

//Take a function 
  Eigen::VectorXf f_w(pc_s.Area());
  // for (int i=0; i<pc_s.Area(); ++i){
  //   // f_w(i) = pc_s_spherical[i](0);
  //   f_w(i) = pc_s[i](1);
  // }
  tdp::f_indicator(pc_s, dIndices, f_w);
  std::cout << "checking f indicator f_w" << std::endl;
  for (auto i = dIndices.begin(); i != dIndices.end(); ++i){
    std::cout << f_w(*i) << " ";
  }
  std::cout << std::endl;
  if ( dIndices.size() <= 0){
    f_w.fill(1);
  }

//To local
  Eigen::VectorXf f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
  Eigen::VectorXf d_f_l = D * f_l;
  Eigen::VectorXf d_f_w = S_wl * d_f_l;
  t0.toctic("diff operation");


  std::cout << "--------FINAL RESULT---------" << std::endl;
  std::cout << "max_phi: " << (float)max_phi  << std::endl;
  std::cout << (f_w-d_f_w).norm() << std::endl;
  //todo change max_phi and should get zero here
    std::cout << (f_w.squaredNorm()/d_f_w.dot(f_w)) << std::endl;

  std::cout << "AY YO!" << std::endl;
  return 0;
}

