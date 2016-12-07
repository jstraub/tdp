/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
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


#include <tdp/laplace_beltrami/laplace_beltrami.h>

//Given C mtx, x (index to pc_s, Source Manifold) , find y (index to pc_t) the correspondence in the Target Manifold
int getCorrespondence(const tdp::Image<tdp::Vector3fda>& pc_s,
                      const tdp::Image<tdp::Vector3fda>& pc_t,
                      const Eigen::MatrixXf& S_wl,
                      const Eigen::MatrixXf& T_wl,
                      const Eigen::MatrixXf& C,
                      const float alpha,
                      const int qId){
    Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area());
    Eigen::VectorXf f_l(C.rows()), g_l(C.rows());
    int target_r, target_c;
    tdp::f_rbf(pc_s, pc_s[qId], alpha, f_w);
    f_l = (f_w.transpose()*S_wl).transpose();
    g_l = C*f_l;
    g_w = T_wl*g_l;
    g_w.maxCoeff(&target_r,&target_c);
    return target_r;
}


void Test_simplePc(){
    tdp::ManagedHostImage<tdp::Vector3fda> pc_s = tdp::GetSimplePc();
    tdp::ManagedHostImage<tdp::Vector3fda> pc_t = tdp::GetSimplePc();
    // parameters
    int numEv = pc_s.Area()/2;//pc_s.Area()-2; //get ALL eigenvectors of L
    int knn = pc_s.Area(); // use all points as neighbors
    float eps = 1e-6;
    float alpha = 0.01;
    float alpha2 = 0.1;

    int numCst = numEv;//pc_s.Area();
    int numQ = pc_s.Area();
    // build kd tree
    tdp::ANN ann_s, ann_t;
    ann_s.ComputeKDtree(pc_s);
    ann_t.ComputeKDtree(pc_t);

    // construct laplacian matrices
    Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area());
    Eigen::SparseMatrix<float> L_t(pc_t.Area(), pc_t.Area());
    Eigen::MatrixXf S_wl(L_s.rows(),(int)numEv);//cols are evectors
    Eigen::MatrixXf T_wl(L_t.rows(),(int)numEv);

    L_s = tdp::getLaplacian(pc_s, ann_s, knn, eps, alpha);
    L_t = tdp::getLaplacian(pc_t, ann_t, knn, eps, alpha);
    tdp::getLaplacianBasis(L_s, numEv, S_wl);
    tdp::getLaplacianBasis(L_t, numEv, T_wl);

    std::cout << "Basis ---" << std::endl;
    std::cout << S_wl << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << T_wl << std::endl;

    //--playing around here
    Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area());
    Eigen::VectorXf f_l((int)numEv), g_l((int)numEv);
    Eigen::MatrixXf F((int)numCst, (int)numEv), G((int)numCst, (int)numEv);
    Eigen::MatrixXf C((int)numEv, (int)numEv);

    // --construct F(design matrix) using point correspondences

    for (int i=0; i< (int)numCst; ++i){

        tdp::f_rbf(pc_s, pc_s[i], alpha2, f_w); //todo: check if I can use this same alpha?
        tdp::f_rbf(pc_t, pc_t[i], alpha2, g_w);
        f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
        g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);

        F.row(i) = f_l;
        G.row(i) = g_l;
    }

    // solve least-square
    C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);
    //std::cout << "F: \n" << F.rows() << F.cols() << std::endl;
    //std::cout << "\nG: \n" << G.rows() << G.cols() << std::endl;
    std::cout << "\nC: \n" << C << /*C.rows() << C.cols() <<*/ std::endl;

    //Get correspondences
    Eigen::VectorXi nnIds(1);
    Eigen::VectorXf dists(1);
    tdp::ManagedHostImage<tdp::Vector3fda> queries((int)numQ,1);
    tdp::ManagedHostImage<tdp::Vector3fda> estimates((int)numQ,1);
    for (int i=0; i<(int)numQ; ++i){
        int tId = getCorrespondence(pc_s, pc_t, S_wl, T_wl, C, alpha2, i); //guessed id in second manifold
        ann_t.Search(pc_s[i], 1, 1e-9, nnIds, dists);
        queries[i] = pc_s[i];
        estimates[i] = pc_t[tId];
        std::cout << "query: \n" << pc_s[i].transpose()<<std::endl;
        std::cout << "guess: \n" << pc_t[tId].transpose() << std::endl;
        std::cout << "true: \n" << pc_t[nnIds(0)].transpose() << std::endl;
    }


}



int main(int argc, char* argv[]){
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
  pangolin::OpenGlRenderState t_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  pangolin::OpenGlRenderState cmtx_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewPc);
  pangolin::View& viewPc_t = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(t_cam));
  container.AddDisplay(viewPc_t);
  pangolin::View& viewCMtx = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(cmtx_cam));
  container.AddDisplay(viewCMtx);
  pangolin::View& viewF = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewF);
  pangolin::View& viewG = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(t_cam));
  container.AddDisplay(viewG);

  // Add variables to pangolin GUI
  pangolin::Var<bool> showFMap("ui.show fMap", true, false);
  pangolin::Var<bool> showMeans("ui.show means", true, false);
  pangolin::Var<bool> runQuery("ui.run queries", true, false);
  pangolin::Var<bool> showFTransfer(" ui. show fTransfer", true, true);

  pangolin::Var<int> nSamples("ui. num samples from mesh pc", 1000, 1000, 10000);

  // pangolin::Var<int> pcOption("ui. pc option", 0, 0,1);
  //-- variables for KNN
  pangolin::Var<int> knn("ui.knn",30,1,100);
  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);
  pangolin::Var<float> alpha("ui. alpha", 0.01, 0.001, 0.3); //variance of rbf kernel for laplacian
  pangolin::Var<float> alpha2("ui. alpha2", 0.001, 0.001, 0.5); //variance of rbf kernel for defining function on manifold
  pangolin::Var<int> nBins("ui. nBins", 10, 10,100);
  //--Correspondence Matrix C estimation
  pangolin::Var<int> numEv("ui.numEv",50,10,300); //min=1, max=pc_s.Area()
  pangolin::Var<int> numCst("ui.numCst",2*numEv/*std::min(20*numEv, pc_s.Area())*/,numEv,nSamples);
  //-- viz color coding
  pangolin::Var<float>minVal("ui. min Val",-0.71,-1,0);
  pangolin::Var<float>maxVal("ui. max Val",0.01,1,0);
  pangolin::Var<int>numQ("ui. num Queries", 100, 100, nSamples);
  float minVal_t, maxVal_t, minVal_c, maxVal_c, minCValue, maxCValue,
          minF0Value, maxF0Value, minG0Value, maxG0Value;

  //End of Pangoline GUI setup

  //Test_simplePc();
  //return 1;
  tdp::ManagedHostImage<tdp::Vector3fda> pc_s(nSamples,1);
  tdp::ManagedHostImage<tdp::Vector3fda> ns_s(nSamples,1);

  tdp::ManagedHostImage<tdp::Vector3fda> pc_t(nSamples,1);
  tdp::ManagedHostImage<tdp::Vector3fda> ns_t(nSamples,1);

  tdp::ManagedHostImage<tdp::Vector3fda> pc_all;
  tdp::ManagedHostImage<tdp::Vector3uda> trigs_all;

  // Visualization of C
  tdp::ManagedHostImage<tdp::Vector3fda> pc_grid((int)numEv*(int)numEv,1);

  if(argc > 1){
      const std::string input = std::string(argv[1]);
      tdp::LoadPointCloudFromMesh(input, pc_all);
      //std::cout << "input pc: " << input << std::endl;
      //std::cout << "triangle meshs loaded. Num points:  " << pc_all.Area() << std::endl;
//      getSamples(pc_all, pc_s, nSamples);
//      getSamples(pc_all, pc_t, nSamples);
  }else{
      std::srand(101);
      GetSphericalPc(pc_s, nSamples);
//      std::srand(200);
      std::srand(101);
      GetSphericalPc(pc_t, nSamples);
  }

    // build kd tree
//  tdp::ANN ann_s, ann_t;
//  ann_s.ComputeKDtree(pc_s);
//  ann_t.ComputeKDtree(pc_t);

  // use those OpenGL buffers
  pangolin::GlBuffer vbo, vbo_t, vbo_cmtx,
                     vbo_f, vbo_g,  //point clouds
                     vbo_f0, vbo_g0,
                     valuebo_f, valuebo_g, valuebo_cmtx, valuebo_color, //colorings: source manifold, target manifod, c_mtx
                     valuebo_f0, valuebo_g0;

  //-- upload point cloud positions
//  vbo.Reinitialise(pangolin::GlArrayBuffer, pc_s.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
//  vbo.Upload(pc_s.ptr_, pc_s.SizeBytes(), 0);
//  vbo_t.Reinitialise(pangolin::GlArrayBuffer, pc_t.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
//  vbo_t.Upload(pc_t.ptr_, pc_t.SizeBytes(), 0);
//  vbo_cmtx.Reinitialise(pangolin::GlArrayBuffer, pc_grid.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
//  vbo_cmtx.Upload(pc_grid.ptr_, pc_grid.SizeBytes(), 0);

  // Declare variables
  Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area()),
                             L_t(pc_t.Area(), pc_t.Area());
  Eigen::VectorXf evector_s,//(L_s.rows());
                  evector_t;//(L_t.rows());
  Eigen::MatrixXf S_wl,//(L_s.rows(),(int)numEv),
                  T_wl,//(L_t.rows(),(int)numEv),
                  F,//((int)numCst, (int)numEv),
                  G,//((int)numCst, (int)numEv),
                  C;//((int)numEv, (int)numEv);
  tdp::eigen_vector<tdp::Vector3fda> means_s((int)nBins, tdp::Vector3fda(0,0,0)),
                                     means_t((int)nBins, tdp::Vector3fda(0,0,0));

//  GetGrid(pc_grid, (int)numEv, (int)numEv);
//  std::cout << "pc mtx: "<< std::endl;
//  for (int r=0; r<(int)numEv; ++r){
//      std::cout << "r: " << r << std::endl;
//      for (int c=0; c<(int)numEv; ++c){
//          std::cout << pc_grid[r*(int)numEv + c] << std::endl;
//      }
//      std::cout << std::endl;
//  }

  //---Queries
  Eigen::VectorXi nnIds(1);
  Eigen::VectorXf dists(1);
  Eigen::VectorXi qIds;
  Eigen::VectorXf qDists;
  tdp::ManagedHostImage<tdp::Vector3fda> queries;
  tdp::ManagedHostImage<tdp::Vector3fda> estimates;
  tdp::ANN ann_s, ann_t;
  //----color scheme
  Eigen::VectorXf colors;//((int)numQ);

  // Control switches
  bool annChanged = false;
  bool laplacianChanged = false;
  bool basisChanged = false;
  bool cChanged = false;
  bool queryChanged = false;

  // Stream and display video
  while(!pangolin::ShouldQuit()){
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Get samples
    if ( pangolin::Pushed(showFMap) || nSamples.GuiChanged() ){
        std::cout << "Running fMap from top..." << std::endl;

        if (argc>1){
            tdp::GetSamples(pc_all, pc_s, nSamples);
            tdp::GetSamples(pc_all, pc_t, nSamples);
        } else{
            std::srand(101);
            GetSphericalPc(pc_s, nSamples);
            std::srand(101);
            GetSphericalPc(pc_t, nSamples);
        }

        vbo.Reinitialise(pangolin::GlArrayBuffer, pc_s.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc_s.ptr_, pc_s.SizeBytes(), 0);
        vbo_t.Reinitialise(pangolin::GlArrayBuffer, pc_t.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo_t.Upload(pc_t.ptr_, pc_t.SizeBytes(), 0);

        // build kd tree
        ann_s.ComputeKDtree(pc_s);
        ann_t.ComputeKDtree(pc_t);

        annChanged = true;
        std::cout << "ANN changed" << std::endl;
    }

    if (pangolin::Pushed(showFMap) || annChanged ||
                  knn.GuiChanged() || alpha.GuiChanged()){
        // get Laplacian operator and its eigenvectors
        tdp::Timer t0;
        L_s = tdp::getLaplacian(pc_s, ann_s, knn, eps, alpha);
        L_t = tdp::getLaplacian(pc_t, ann_t, knn, eps, alpha);
        std::cout << "L_s and L_t changed: " << alpha << std::endl;
        t0.toctic("GetLaplacians");
        laplacianChanged = true;
        std::cout << "laplacian changed" << std::endl;
    }


    if (pangolin::Pushed(showFMap) || laplacianChanged ||
                numEv.GuiChanged() || nBins.GuiChanged()){
      S_wl.resize(L_s.rows(),(int)numEv);
      T_wl.resize(L_t.rows(),(int)numEv);
      std::cout << "s_wl resized: " << S_wl.rows() << ", " << S_wl.cols() << std::endl;
      std::cout << "t_wl resized: " << T_wl.rows() << ", " << T_wl.cols() << std::endl;

      tdp::Timer t0;
      tdp::getLaplacianBasis(L_s, numEv, S_wl);
      evector_s = S_wl.col(1); // first non-trivial evector
      means_s = tdp::getLevelSetMeans(pc_s, evector_s, (int)nBins); //means based on the evector_s's nBins level sets

      tdp::getLaplacianBasis(L_t, numEv, T_wl);
      evector_t = T_wl.col(1); // first non-trivial evector
      means_t = tdp::getLevelSetMeans(pc_t, evector_t, (int)nBins);
      t0.toctic("GetEigenVectors & GetMeans");

      valuebo_f.Reinitialise(pangolin::GlArrayBuffer, evector_s.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_f.Upload(&evector_s(0), sizeof(float)*evector_s.rows(), 0);
      std::cout << evector_s.minCoeff() << " " << evector_s.maxCoeff() << std::endl;
      minVal = evector_s.minCoeff()-1e-3;
      maxVal = evector_s.maxCoeff();

      valuebo_g.Reinitialise(pangolin::GlArrayBuffer, evector_t.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_g.Upload(&evector_t(0), sizeof(float)*evector_t.rows(), 0);
      std::cout << evector_t.minCoeff() << " " << evector_t.maxCoeff() << std::endl;
      minVal_t = evector_t.minCoeff()-1e-3;
      maxVal_t = evector_t.maxCoeff();

      basisChanged = true;
      std::cout << "basisChanged" << std::endl;
      }

    if (pangolin::Pushed(showFMap) || basisChanged ||
                numCst.GuiChanged()|| alpha2.GuiChanged()){

      F.resize((int)numCst, (int)numEv);
      G.resize((int)numCst, (int)numEv);
      C.resize((int)numEv, (int)numEv);
      Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area());
      Eigen::VectorXf f_l((int)numEv), g_l((int)numEv);

      // Automatic correspondence construction using one point
      // as the center and getting numCst (closest) points around it
      Eigen::VectorXi sIds((int)numCst), tIds((int)numCst);
      Eigen::VectorXf sDists((int)numCst), tDists((int)numCst);
      ann_s.Search(pc_s[0], (int)numCst, 1e-9, sIds, sDists);
      ann_t.Search(pc_t[0], (int)numCst, 1e-9, tIds, tDists);
      // --construct F(design matrix) using point correspondences
      for (int i=0; i<(int)numCst; ++i){
          //ann_t.Search(pc_s[i], 1, 1e-9, nnIds, dists);
          //std::cout << "match idx: " << i << ", " << nnIds(0) << std::endl;
          //tdp::f_rbf(pc_s, pc_s[i], alpha, f_w); //todo: check if I can use this same alpha?
          //tdp::f_rbf(pc_t, pc_t[nnIds(0)], alpha, g_w);
          //std::cout << "f_w: " /*<< f_w.transpose() */<< std::endl;
          //std::cout << "g_w: " /*<< g_w.transpose()*/ << std::endl;

          //f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
          //std::cout << "f_l: " << f_l.transpose() << std::endl;
          //g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);
          //std::cout << "g_l: " << g_l.transpose() << std::endl;

          tdp::f_rbf(pc_s, pc_s[sIds[i]], alpha2, f_w); //todo: check if I can use this same alpha2?
          tdp::f_rbf(pc_t, pc_t[tIds[i]], alpha2, g_w);
          f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
          g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);

          F.row(i) = f_l;
          G.row(i) = g_l;
      }
      // solve least-square
      C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);

      //Visualization of C
      GetGrid(pc_grid, (int)numEv, (int)numEv);
      vbo_cmtx.Reinitialise(pangolin::GlArrayBuffer, pc_grid.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_cmtx.Upload(pc_grid.ptr_, pc_grid.SizeBytes(), 0);

      //color coding of the C matrix
      Eigen::VectorXf cvec((int)numEv*(int)numEv);
      for (int r=0; r<C.rows(); ++r){
          for (int c=0; c<C.cols(); ++c){
              cvec(r*C.cols()+c) = C(r,c);
          }
      }

      valuebo_cmtx.Reinitialise(pangolin::GlArrayBuffer, cvec.rows(), GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_cmtx.Upload(&cvec(0), sizeof(float)*cvec.rows(), 0);

      std::cout << cvec.minCoeff() << " " << cvec.maxCoeff() << std::endl;
      minVal_c = cvec.minCoeff()-1e-3;
      maxVal_c = cvec.maxCoeff();

      cChanged = true;
      }

      // Get the point-wise correspondence
      //--Query 100 closest points to pc_s[0]
      //--The result points should also be close to each other
    if ( (viewF.IsShown() && viewG.IsShown()) &&
         (pangolin::Pushed(showFMap) || cChanged || numQ.GuiChanged()) ){
        std::cout << "current numQ: " << numQ<< std::endl;

        qIds.resize((int)numQ);
        qDists.resize((int)numQ);
        ann_s.Search(pc_s[0], (int)numQ, 1e-9, qIds, qDists);

        queries.Reinitialise((int)numQ,1);
        estimates.Reinitialise((int)numQ,1);
        for (int i=0; i<(int)numQ; ++i){
            //todo: random i
            int tId = getCorrespondence(pc_s, pc_t, S_wl, T_wl, C, alpha2, i); //guessed id in second manifold
            ann_t.Search(pc_s[qIds[i]], 1, 1e-9, nnIds, dists);

            queries[i] = pc_s[qIds[i]];
            estimates[i] = pc_t[tId];
            //          std::cout << "query: \n" << pc_s[qIds[i]]<<std::endl;
            //          std::cout << "guess: \n" << pc_t[tId] << std::endl;
            //          std::cout << "true: \n" << pc_t[nnIds(0)] << std::endl;
        }

        //--visualization
        std::cout << "before: " << colors.rows() << std::endl;
        colors.resize((int)numQ);
        for (int i=0; i<(int)numQ; ++i){
            colors(i) = (i*0.001);
        }
        minCValue = colors.minCoeff()-1e-3;
        maxCValue = colors.maxCoeff();
        std::cout << "mincolor: " << minCValue << std::endl;
        std::cout << "maxcolor: " << maxCValue << std::endl;

        vbo_f.Reinitialise(pangolin::GlArrayBuffer, queries.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo_f.Upload(queries.ptr_, queries.SizeBytes(), 0);

        vbo_g.Reinitialise(pangolin::GlArrayBuffer, estimates.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo_g.Upload(estimates.ptr_, estimates.SizeBytes(), 0);

        valuebo_color.Reinitialise(pangolin::GlArrayBuffer, colors.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
        valuebo_color.Upload(&colors(0), sizeof(float)*colors.rows(), 0);

        //TODO: segment correspondence
        //    : operator commutativity constraint?
        //    : any other constraint to add to the least square?
        //TODO: add a handler to choose correponding points from each point cloud by clicking on the gui
        //    : use the points to construct least-square
        //    : check how well that works
        //START HERE!
        //--end of playing
        queryChanged = true;
        std::cout << "<--DONE fMap-->" << std::endl;
    }
    if (queryChanged && showFTransfer){
        std::cout << "showFTransfer: " << showFTransfer << std::endl;

        //Show function transfer
        //--pick a point 10
        Eigen::VectorXf f0_w(pc_s.Area()), g0_w(pc_t.Area());
        Eigen::VectorXf f0_l((int)numEv), g0_l((int)numEv);

        tdp::f_rbf(pc_s, pc_s[0], alpha2, f0_w);
        f0_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f0_w);
        g0_l = C*f0_l;
        g0_w = T_wl*g0_l;

        // Alternative
        //int tId = getCorrespondence(pc_s, pc_t, S_wl, T_wl, C, alpha2, i); //guessed id in second manifold
        //tdp::f_rbf(pc_t, pc_t[tId], alpha2, g0_w);

        vbo_f0.Reinitialise(pangolin::GlArrayBuffer, pc_s.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo.Upload(pc_s.ptr_, pc_s.SizeBytes(), 0);
        vbo_t.Reinitialise(pangolin::GlArrayBuffer, pc_t.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo_t.Upload(pc_t.ptr_, pc_t.SizeBytes(), 0);

        valuebo_f0.Reinitialise(pangolin::GlArrayBuffer, f0_w.rows(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        valuebo_f0.Upload(&f0_w(0), sizeof(float)*f0_w.rows(), 0);

        valuebo_g0.Reinitialise(pangolin::GlArrayBuffer, g0_w.rows(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        valuebo_g0.Upload(&g0_w(0), sizeof(float)*g0_w.rows(), 0);

        // Coloring range
        minF0Value = f0_w.minCoeff()-1e-3;
        maxF0Value = f0_w.maxCoeff();
        minG0Value = g0_w.minCoeff()-1e-3;
        maxG0Value = g0_w.maxCoeff();
        std::cout << "minF0: " << minF0Value << std::endl;
        std::cout << "maxF0: " << maxF0Value << std::endl;
        std::cout << "minG0: " << minG0Value << std::endl;
        std::cout << "maxG0: " << maxG0Value << std::endl;
        //std::cout << "f vals: " << f0_w.transpose() << std::endl;
        std::cout << "<--DONE fMap-->" << std::endl;

    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (viewPc.IsShown()) {
      viewPc.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minVal);
      shader.SetUniform("maxValue", maxVal);
      valuebo_f.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
      vbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      valuebo_f.Unbind();
      glDisableVertexAttribArray(0);
      vbo.Unbind();


      // draw lines connecting the means
      if (showMeans){
          glColor3f(.3,1.,.125);
          glLineWidth(2);
          tdp::Vector3fda m, m_prev;
          for (size_t i=1; i<means_s.size(); ++i){
              m_prev = means_s[i-1];
              m = means_s[i];
              tdp::glDrawLine(m_prev, m);
          }
      }
    }

    if (viewPc_t.IsShown()){
        viewPc_t.Activate(t_cam);
        pangolin::glDrawAxis(0.1);

        glPointSize(2.);
        glColor3f(1.0f, 1.0f, 0.0f);
        // renders the vbo with colors from valuebo
        auto& shader_t = tdp::Shaders::Instance()->valueShader_;
        shader_t.Bind();
        shader_t.SetUniform("P",  t_cam.GetProjectionMatrix());
        shader_t.SetUniform("MV", t_cam.GetModelViewMatrix());
        shader_t.SetUniform("minValue", minVal_t);
        shader_t.SetUniform("maxValue", maxVal_t);
        valuebo_g.Bind();
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        vbo_t.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(4.);
        glDrawArrays(GL_POINTS, 0, vbo_t.num_elements);
        shader_t.Unbind();
        glDisableVertexAttribArray(1);
        valuebo_g.Unbind();
        glDisableVertexAttribArray(0);
        vbo_t.Unbind();

        // draw lines connecting the means
        if (showMeans){
            glColor3f(.3,1.,.125);
            glLineWidth(2);
            tdp::Vector3fda m, m_prev;
            for (size_t i=1; i<means_t.size(); ++i){
                m_prev = means_t[i-1];
                m = means_t[i];
                tdp::glDrawLine(m_prev, m);
            }
        }
    }

    if (viewCMtx.IsShown()){
        viewCMtx.Activate(cmtx_cam);
        pangolin::glDrawAxis(0.1);

        // plots dots with the same number of rows and cols of C
        glPointSize(5.);
        glColor3f(1.0f, 1.0f, 0.0f);

        // renders the vbo with colors from valuebo
        auto& shader = tdp::Shaders::Instance()->valueShader_;
        shader.Bind();
        shader.SetUniform("P",  cmtx_cam.GetProjectionMatrix());
        shader.SetUniform("MV", cmtx_cam.GetModelViewMatrix());
        shader.SetUniform("minValue", minVal_c);
        shader.SetUniform("maxValue", maxVal_c);
        valuebo_cmtx.Bind();
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        vbo_cmtx.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(4.);
        glDrawArrays(GL_POINTS, 0, vbo_cmtx.num_elements);
        shader.Unbind();
        glDisableVertexAttribArray(1);
        valuebo_cmtx.Unbind();
        glDisableVertexAttribArray(0);
        vbo_cmtx.Unbind();

    }

    if (viewF.IsShown() && (!showFTransfer) ){

//        viewF.Activate(s_cam);
//        pangolin::glDrawAxis(0.1);

//        // draw lines connecting the means
//        glColor3f(.3,1.,.125);
//        glLineWidth(2);
//        tdp::Vector3fda m, m_prev;
//        for (size_t i=1; i<means_s.size(); ++i){
//            m_prev = means_s[i-1];
//            m = means_s[i];
//            tdp::glDrawLine(m_prev, m);
//        }

//        glPointSize(2.);
//        glColor3f(1.0f, 1.0f, 0.0f);
//        // renders the vbo with colors from valuebo
//        auto& shader = tdp::Shaders::Instance()->valueShader_;
//        shader.Bind();
//        shader.SetUniform("P",  s_cam.GetProjectionMatrix());
//        shader.SetUniform("MV", s_cam.GetModelViewMatrix());
//        shader.SetUniform("minValue", minVal);
//        shader.SetUniform("maxValue", maxVal);
//        valuebo_s.Bind();
//        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
//        vbo.Bind();
//        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

//        glEnableVertexAttribArray(0);
//        glEnableVertexAttribArray(1);
//        glPointSize(4.);
//        glDrawArrays(GL_POINTS, 0, vbo.num_elements);
//        shader.Unbind();
//        glDisableVertexAttribArray(1);
//        valuebo_s.Unbind();
//        glDisableVertexAttribArray(0);
//        vbo.Unbind();


        viewF.Activate(s_cam);
        pangolin::glDrawAxis(0.1);

        glPointSize(2.);
        glColor3f(1.0f, 1.0f, 0.0f);
        // renders the vbo with colors from valuebo
        auto& shader = tdp::Shaders::Instance()->valueShader_;
        shader.Bind();
        shader.SetUniform("P",  s_cam.GetProjectionMatrix());
        shader.SetUniform("MV", s_cam.GetModelViewMatrix());
        shader.SetUniform("minValue", minCValue);
        shader.SetUniform("maxValue", maxCValue);
        valuebo_color.Bind();
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        vbo_f.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(4.);
        glDrawArrays(GL_POINTS, 0, vbo_f.num_elements);
        shader.Unbind();
        glDisableVertexAttribArray(1);
        valuebo_color.Unbind();
        glDisableVertexAttribArray(0);
        vbo_f.Unbind();
    }

    if (viewG.IsShown() && (!showFTransfer) ){

        viewG.Activate(t_cam);
        pangolin::glDrawAxis(0.1);

        glPointSize(2.);
        glColor3f(1.0f, 1.0f, 0.0f);
        // renders the vbo with colors from valuebo
        auto& shader = tdp::Shaders::Instance()->valueShader_;
        shader.Bind();
        shader.SetUniform("P",  t_cam.GetProjectionMatrix());
        shader.SetUniform("MV", t_cam.GetModelViewMatrix());
        shader.SetUniform("minValue", minCValue);
        shader.SetUniform("maxValue", maxCValue);
        valuebo_color.Bind();
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        vbo_g.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(4.);
        glDrawArrays(GL_POINTS, 0, vbo_g.num_elements);
        shader.Unbind();
        glDisableVertexAttribArray(1);
        valuebo_color.Unbind();
        glDisableVertexAttribArray(0);
        vbo_g.Unbind();
    }

    if (viewF.IsShown()&& showFTransfer) {
      viewF.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minF0Value);
      shader.SetUniform("maxValue", maxF0Value);
      //(showFTransfer)? valuebo_f0.Bind(): valuebo_f.Bind();
      valuebo_f0.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
      vbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      //(showFTransfer)? valuebo_f0.Unbind(): valuebo_f.Unbind();
      valuebo_f0.Unbind();
      glDisableVertexAttribArray(0);
      vbo.Unbind();

    }

    if (viewG.IsShown() && showFTransfer){
        viewG.Activate(t_cam);
        pangolin::glDrawAxis(0.1);

        glPointSize(2.);
        glColor3f(1.0f, 1.0f, 0.0f);
        // renders the vbo with colors from valuebo
        auto& shader_t = tdp::Shaders::Instance()->valueShader_;
        shader_t.Bind();
        shader_t.SetUniform("P",  t_cam.GetProjectionMatrix());
        shader_t.SetUniform("MV", t_cam.GetModelViewMatrix());
        shader_t.SetUniform("minValue", minG0Value);
        shader_t.SetUniform("maxValue", maxG0Value);
        valuebo_g0.Bind();
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        vbo_t.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(4.);
        glDrawArrays(GL_POINTS, 0, vbo_t.num_elements);
        shader_t.Unbind();
        glDisableVertexAttribArray(1);
        valuebo_g0.Unbind();
        glDisableVertexAttribArray(0);
        vbo_t.Unbind();
    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
    // reset some switches
    annChanged = false;
    laplacianChanged = false;
    basisChanged = false;
    cChanged = false;
    queryChanged = false;
  }

  std::cout << "good morning!" << std::endl;
  return 0;
}
