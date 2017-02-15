#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <random>

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


/************TODO***********************************************/
/***************************************************************/
//1. MOVE THE TESTS TO A SEPARATE TEST FILE
//2. FOCUS ON arm.ply and bunny ply

/************Declarations***************************************
 ***************************************************************/
void Test_samePc_sameSamples(
        int nSamples, 
        float noiseStd,
        int nEv,
        float factor,
        int nHKS,
        float thresh,
        std::string& option,
        int shapeOpt,
        bool showDecomposition = false);

int main(int argc, char* argv[]){
  std::cout << "Test Correspondences---" << std::endl;
  std::string option("rbf");
  int shapeOpt = 2;
  int nSamples = 450;
  int nEv = 30;
  float factor = 8.0f;// = 1.0f; //to decide nPW = ceil(factor*nEv);
  int nHKS = 0;
  float noiseStd = 0;
  float thresh(0.f); //threshold for cmatrix pad to zero or one
  bool showDecomposition = false;
  std::vector<float> thresholds{0.0f, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};
 
  Test_samePc_sameSamples(nSamples, noiseStd, nEv, factor, 
                          nHKS,
                          thresh,
                          option,
                          shapeOpt,
                          showDecomposition);


  // std::cout << "Experiment-------------------------" <<std::endl;
  // std::cout << "shape: " << shapeOpt << ", nPoints: " << nSamples 
  //           << ", nEv: " << nEv  << std::endl; 

  // for (int i =0; i<thresholds.size(); ++i){
  //   std::cout << "\n================================="<< std::endl;
  //   std::cout << i << ". threshold: " << thresholds[i] << std::endl;
  //   for (int f=0; f< floor(nSamples/nEv); ++f){
  //       std::cout << "\tfactor(nEv->nPW): " << f << std::endl;

  //       Test_samePc_sameSamples(nSamples, noiseStd, nEv, (float)f, nHKS, thresholds[i],
  //                             option, shapeOpt, showDecomposition);
  //       std::cout << "=================================\n"<< std::endl;
  //   }
  // } 

}


// void Test_samePc_withGNoise
void Test_samePc_sameSamples(int nSamples, float noiseStd,
                            int nEv, float factor, int nHKS,
                            float thresh,
                            std::string& option,
                            int shapeOpt,
                            bool showDecomposition){
  /* nSamples: number of points on the surface
   * nTrain: number of  correpondence pairs given to approximate functional mapping
   *  - must be at least numEv, and at most nSamples
   * option: "rbf" or "ind".  function to construct a function on the surface based on a given point
   * shapeOpt: 0 is "linear" for simplePc,
   *         : 1 is "sphere" for random sampling from unit sphere
   *         : 2 is "bunny",
   *         : 3 is "manekine"
   * showDecompotion: true to see (eigenfunctions and) eigenvalues of the Laplacian of both surfaces
   */

//todo: std::option
    tdp::ManagedHostImage<tdp::Vector3fda> pc_all;
    tdp::ManagedHostImage<tdp::Vector3fda> pc_s;
    tdp::ManagedHostImage<tdp::Vector3fda> pc_t;
    std::string fpath;

    // Get point clouds
    switch (shapeOpt){
      case 0:
        tdp::GetSimplePc(pc_s, nSamples);
        std::cout << "Shape: linear---" << std::endl;
        break;
      case 1:
        tdp::GetSphericalPc(pc_s, nSamples);
        std::cout << "Shape: sphere---" << std::endl;
        break;
      case 2:
        fpath = std::string("/home/hjsong/workspace/data/mesh/bun_zipper_res4.ply"); //todo: hardcoded
        tdp::LoadPointCloudFromMesh(fpath, pc_all);
        tdp::GetSamples(pc_all, pc_s, nSamples);
        std::cout << "Shape: bunny---" << std::endl;
        break;
      case 3:
        fpath = std::string("/home/hjsong/workspace/data/mesh/cleanCylinder_0.ply"); //todo: hardcoded
        tdp::LoadPointCloudFromMesh(fpath, pc_all);
        tdp::GetSamples(pc_all, pc_s, nSamples);
        std::cout << "Shape: manekine---" << std::endl;
    }

    // pc_t.ResizeCopyFrom(pc_s);
    addGaussianNoise(pc_s, (float)noiseStd, pc_t);

    //tdp::printImage(pc_s, 0, pc_s.Area());


    /****parameters*****/
    /*******************/
    //int nEv = std::min(30, (int)(pc_s.Area()/3));//pc_s.Area()-2;
    //get ALL eigenvectors of L
    int knn = pc_s.Area(); // use all points as neighbors
    float eps = 1e-6;
    float alpha = 0.01;
    float alpha2 = 0.001;

    int nPW = ceil(factor*nEv);//number of pointwise correspondences
    assert(nPW < pc_s.Area());

    // int nHKS = 0; //number of heat kernel signature correspondences
    int nCst = nPW + nHKS;//pc_s.Area();
    
    //int nTest = pc_s.Area() - nPW;
    int nShow = std::min(10, nEv);
    /*******************/

    // build kd tree
    tdp::ANN ann_s, ann_t;
    ann_s.ComputeKDtree(pc_s);
    ann_t.ComputeKDtree(pc_t);

    // construct laplacian matrices
    Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area()),
                               L_t(pc_t.Area(), pc_t.Area());
    Eigen::MatrixXf S_wl(L_s.rows(),(int)nEv),//cols are evectors
                    T_wl(L_t.rows(),(int)nEv),
                    S_desc_w, T_desc_w,
                    S_desc_l, T_desc_l;
    Eigen::VectorXf S_evals((int)nEv), T_evals((int)nEv);

    //******************CACHE NAMING*************************//
    //*******************************************************//
    std::map<std::string, std::string> cacheDic = tdp::makeCacheNames(
                              (int)shapeOpt, (int)nSamples, (int)knn, 
                              (float)alpha, (float)noiseStd, (int)nEv);
    const char* path_ls = cacheDic.at("ls").c_str();
    const char* path_lt = cacheDic.at("lt").c_str();    
    const char* path_s_wl = cacheDic.at("s_wl").c_str();
    const char* path_t_wl = cacheDic.at("t_wl").c_str();
    const char* path_s_evals = cacheDic.at("s_evals").c_str();
    const char* path_t_evals = cacheDic.at("t_evals").c_str();    

    int res = access(path_ls, R_OK) + access(path_lt, R_OK);
    if (res == 0){
        // Read cached file
        std::cout << "Reading Laplacians from cache---" << std::endl;
        
        tdp::read_binary(path_ls, L_s);
        tdp::read_binary(path_lt, L_t);
    } else{
        L_s = tdp::getLaplacian(pc_s, ann_s, knn, eps, alpha);
        L_t = tdp::getLaplacian(pc_t, ann_t, knn, eps, alpha);

        tdp::write_binary(path_ls, L_s);
        tdp::write_binary(path_lt, L_t);
    
        std::cout << "Cached: Laplacians" << std::endl;
    }

    res = access(path_s_wl, R_OK)  + access(path_t_wl, R_OK) +
          access(path_s_evals, R_OK) + access(path_t_evals, R_OK);
    if (res == 0){    
        std::cout << "Reading Bases&evals from cache---" << std::endl;
        tdp::read_binary(path_s_wl, S_wl);
        tdp::read_binary(path_t_wl, T_wl);
        tdp::read_binary(path_s_evals, S_evals);
        tdp::read_binary(path_t_evals, T_evals);
    } else{
        std::cout << "Calculating Bases&evals---" << std::endl;
        tdp::decomposeLaplacian(L_s, nEv, S_evals, S_wl); //todo: check if size initialization is included
        tdp::decomposeLaplacian(L_t, nEv, T_evals, T_wl);

        tdp::write_binary(path_s_wl, S_wl);
        tdp::write_binary(path_t_wl, T_wl);
        tdp::write_binary(path_s_evals, S_evals);
        tdp::write_binary(path_t_evals, T_evals);
        std::cout << "Cached: bases, evals" << std::endl;
    }

    if (showDecomposition){
        std::cout << "-----------------" << std::endl;
        std::cout << T_wl << std::endl;
        std::cout << "Evals ---" << std::endl;
        std::cout << "\tS: " << S_evals.transpose() << std::endl;
        std::cout << "\tT: " << T_evals.transpose() << std::endl;
    }

    //--Construct function pairs
    Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area()),
                    f_l((int)nEv), g_l((int)nEv);
    Eigen::MatrixXf F((int)nCst, (int)nEv), G((int)nCst, (int)nEv);
    Eigen::MatrixXf C((int)nEv, (int)nEv);

    // --construct F(data matrix) and G based on the correspondences

    for (int i=0; i< (int)nPW; ++i){
      tdp::f_landmark(pc_s, i, alpha2, option, f_w);
      tdp::f_landmark(pc_t, i, alpha2, option, g_w);

      f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
      g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);
      //f_l = tdp::projectToLocal(S_wl, f_w);
      //g_l = tdp::projectToLocal(T_wl, g_w);

      F.row(i) = f_l;
      G.row(i) = g_l;
    }

    if (nHKS >0){
        //-----Add  heat kernel signatures as constraints
        std::cout << "CALCULATEING HKS ---" <<std::endl;
        S_desc_w = tdp::getHKS(S_wl,S_evals,nHKS);
        T_desc_w = tdp::getHKS(T_wl,T_evals,nHKS);
        S_desc_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*S_desc_w);
        T_desc_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*T_desc_w);
        //S_desc_l = tdp::projectToLocal(S_wl, S_desc_w); //columne is a feature
        //T_desc_l = tdp::projectToLocal(T_wl, T_desc_w);
        
        assert(S_desc_l.cols() == nHKS);
        for (int i=0; i<nHKS; ++i){
          F.row(nPW+i) = S_desc_l.col(i);
          G.row(nPW+i) = T_desc_l.col(i);
        }
        std::cout << "S,T descriptors at time 0--------" << std::endl;
        std::cout << S_desc_l.col(0) << std::endl;//heat kernel at timestap i//todo: check at which point for S and T manifolds
        std::cout << T_desc_l.col(0) << std::endl;//heat kernel at timestap i
    }
    //----Add operator constratins
    //

    // solve least-square
    C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);
    float threshold(thresh);  //(1e-5f);
    tdp::clean_near_zero_one(C, threshold);

    int nOnes(0), nZeros(0);
    //Count zeros and ones in the diagnoal of C
    for (int i=0; i< C.rows(); ++i){
      // if (1-threshold <C(i,i) && C(i,i) < 1+threshold){
      if (C(i,i) == 1){
        nOnes += 1;
      } 
      else if (C(i,i) == 0){
        nZeros += 1;
      }
    }

    std::cout << "-----------\n"
              << "C(10x10) \n" 
              << C.block(0,0,nShow,nShow) 
              << std::endl;

    std::cout << "----------\n"
              << "Diagnoals\n"
              << C.diagonal().transpose() 
              << std::endl;
    std::cout << "----------\n"
              << "\t 1: " << nOnes
              << ", 0: " << nZeros
              << std::endl;

    // Test
              //todo: completely new dtaset
              //.    deformed shape, noised shape - restricted region
    int nTest = (int)pc_s.Area()-nPW;
    float error = 0;
    for (int i=nPW; i< (int)pc_s.Area(); ++i ){
        Eigen::VectorXf true_w, true_l, guess_w;
        tdp::f_landmark(pc_s, i, alpha2, option, true_w);
        true_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*true_w);
        guess_w = S_wl * (C*true_l);

        error += (true_w - guess_w).squaredNorm(); //todo:
    }
    error = std::sqrt(error/(nTest));///nSamples; //rms normalized to the vector space dim
    std::cout << "Surface dim: " << pc_s.Area() << std::endl;
    std::cout << "N test points: " << nTest << std::endl;
    std::cout << "rms: " << error << std::endl;

    //transfer z function
    Eigen::VectorXf fz_w(pc_s.Area()), gz_w(pc_t.Area()),
                    fz_l((int)nEv), gz_l((int)nEv);
    for (int i=0; i< pc_s.Area(); ++i){
      fz_w(i) = pc_s[i](2);//*pc_s[i](1);
    }
    fz_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*fz_w);
    gz_l = C*fz_l;
    gz_w = T_wl*gz_l;
    std::cout << "f_l: " << fz_l.transpose() << std::endl;
    std::cout << "g_l: " << gz_l.transpose() << std::endl;
    std::cout << "diff: " << sqrt((fz_l-gz_l).squaredNorm()) << std::endl;
}



