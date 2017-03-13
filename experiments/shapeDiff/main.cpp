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
#include <Eigen/Core>
#include <Eigen/SVD>


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
  //Create openGL window - guess sensible dimensions
  int menu_w = 180;
  pangolin::CreateWindowAndBind("GuiBase", 1200+menu_w, 800);
  //Current frame in memory buffer and displayingh
  pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menu_w));
  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //Setup container
  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
      .SetBounds(0.,1.0, pangolin::Attach::Pix(menu_w), 1.0);
  //Define Camera Render Object (for view/scene browsing)
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
  pangolin::View& view_s = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(s_cam));
  pangolin::View& view_t = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(s_cam));
  pangolin::View& view_cmtx = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(cmtx_cam));
  pangolin::View& view_f = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(s_cam));
  pangolin::View& view_g = pangolin::CreateDisplay()
                           .SetHandler(new pangolin::Handler3D(s_cam));
  pangolin::View& view_d = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(s_cam));

  container.AddDisplay(view_s);
  container.AddDisplay(view_t);
  container.AddDisplay(view_cmtx);
  container.AddDisplay(view_f);
  container.AddDisplay(view_g);
  container.AddDisplay(view_d);


  // Add variables to pangolin GUI
  pangolin::Var<bool> showFMap("ui.show fMap", true, false);
  pangolin::Var<bool> showFTransfer(" ui. show fTransfer", true, true);

  pangolin::Var<int> nSamples("ui. num samples from mesh pc", 2000, 100, 3000);
  pangolin::Var<int> shapeOpt("ui. shape option", 1, 0, 3); //2:bunny

  //--second shape point cloud
  pangolin::Var<float> sFactor("ui. second pc scale", 2.0, 0.5, 3); //pc_t[i] = sFactor*pc_s[i];
  pangolin::Var<float> max_phi("ui. max phi", M_PI/4, 1e-6, M_PI); // spherical deformation using phi angle
  pangolin::Var<float> noiseStd("ui. noiseStd", 0, 0.00001, 0.0001); //zero mean Gaussian noise added to shape S


  //-- variables for KNN
  pangolin::Var<int> knn("ui.knn",30,1,100);//(int)nSamples
  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);
  pangolin::Var<float> alpha("ui. alpha", 0.01, 0.001, 0.3); //variance of rbf kernel for laplacian


  //--Correspondence Matrix C estimation
  pangolin::Var<float> alpha2("ui. alpha2", 0.0001, 0.0001, 0.01); //variance of rbf kernel for defining function on manifold
  pangolin::Var<int> nEv_s("ui.nEv_s",30,30,100); //min=1, max=pc_s.Area()
  pangolin::Var<int> nEv_t("ui.nEv_t",60,30,100); //min=1, max=pc_s.Area()
  pangolin::Var<int> r("ui.Fmap viz scale(r)",4,0,10); //min=1, max=pc_s.Area()


  pangolin::Var<int> nPW("ui.num PointWise train",nSamples/*std::min(20*numEv, pc_s.Area())*/,
                         nEv_t ,nSamples); //todo: minimum = max(nEv_s, nEv_t)

  //-- viz color coding
  pangolin::Var<float>minVal_s("ui. min Val",-0.71,-1,0);
  pangolin::Var<float>maxVal_s("ui. max Val",0.01,1,0);

  // use those OpenGL buffers
  pangolin::GlBuffer vbo_s, vbo_t, vbo_cmtx,

                     valuebo_s, valuebo_t, valuebo_cmtx, valuebo_color, //colorings: source manifold, target manifod, c_mtx
                     valuebo_f, valuebo_g, valuebo_d,
                     valuebo_desc_s, valuebo_desc_t; //shape descriptor values (hks,wks)

  // min,max for coloring
  float minVal_t, maxVal_t, minVal_c, maxVal_c,
        minVal_f, maxVal_f,
        minVal_g, maxVal_g, minVal_d, maxVal_d;

  // Control switches
  bool annChanged = false;
  bool laplacianChanged = false;
  bool basisChanged = false;
  bool cChanged = false;
  bool queryChanged = false;

  /******************End of Pangolin GUI setup*************************/

  //*****************Declare variables*********************************/
  std::string option("rbf");

  tdp::ManagedHostImage<tdp::Vector3fda> 
    pc_all, pc_s_spherical, pc_t_spherical, pc_s, pc_t,
    pc_grid((int)nEv_t*(int)nEv_s,1);//todo s or t first?

  std::string fpath_b, fpath_m;
  // std::map<std::string, std::string> cacheDic;
  tdp::ANN ann_s, ann_t;

  Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area()),//Laplacian of manifold S
                             L_t(pc_t.Area(), pc_t.Area());//Laplacian of manifold T
  Eigen::VectorXf evector_s,//(L_s.rows());
                  evector_t,//(L_t.rows());
                  S_evals,//evalues of Laplacian of S. Increasing order.
                  T_evals;//evalues of Laplacian of T. Increasing order.

  Eigen::MatrixXf S_wl,//(L_s.rows(),(int)numEv),
                  T_wl;//,(L_t.rows(),(int)numEv),
//                  F,//((int)numCst, (int)numEv),
//                  G,//((int)numCst, (int)numEv),
//                  C,//((int)nEv_t, (int)nEv_s);
//                  D;//DiffMap;((int)nEv_t, (int)nEv_s);

  std::vector<int> pIndices; //For functional correpsondence pairs
  std::vector<int> dIndices; //Indices to deformed points

  /*********************************************************************/

  // Stream and display video
  while (!pangolin::ShouldQuit()){
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Get samples
    if ( pangolin::Pushed(showFMap) ||
         nSamples.GuiChanged()      ||
         shapeOpt.GuiChanged()      ||
         noiseStd.GuiChanged()      ||
         sFactor.GuiChanged()       ||
         max_phi.GuiChanged()){
      std::cout << "Running fMap from top..." << std::endl;

      if (argc<3){
          std::cout << "Give paths to bunny and manekine ply files. Terminated." << std::endl;
          return -1;
        } else{
          fpath_b = argv[1];
          fpath_m = argv[2];
        }
        std::cout << "\n\nshape opt: " << shapeOpt << std::endl;

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

            std::cout << "number of deformed: " << dIndices.size() << std::endl;
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

        // Modify second shape
        //todo: pc_s->pc_s_spherical->pc_t_spherical->pc_t
        //tdp::scale(pc_t,sFactor, pc_t); //scale
        //tdp::addGaussianNoise(pc_t, (float)noiseStd, pc_t);

        std::cout << "PC_S: " << pc_s.Area() << std::endl;
        std::cout << "PC_T: " << pc_t.Area() << std::endl;

        //Get indices to be used (later) for correspondence between functions
        std::random_device rd;
        std::mt19937 g(rd());

        for (int i = 0; i<(int)nSamples; ++i){
          pIndices.push_back(i);
        }

        //shuffle correpsondence indices
        std::shuffle(pIndices.begin(), pIndices.end(), g);

        vbo_s.Reinitialise(pangolin::GlArrayBuffer, pc_s.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbo_s.Upload(pc_s.ptr_, pc_s.SizeBytes(), 0);
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
        t0.toctic("GetLaplacians");
        laplacianChanged = true;
        std::cout << "LaplacianS changed\n" << std::endl;
    }

    if (pangolin::Pushed(showFMap) || laplacianChanged ||
                nEv_s.GuiChanged() || nEv_t.GuiChanged() ||
                r.GuiChanged()){
      //nPW = (int)nEv; // + (int)nHKS;
      S_wl.resize(L_s.rows(),(int)nEv_s);
      T_wl.resize(L_t.rows(),(int)nEv_t);
      S_evals.resize((int)nEv_s);
      T_evals.resize((int)nEv_t);

//      F.resize((int)nPW, (int)nEv_s);
//      G.resize((int)nPW, (int)nEv_t);
//      C.resize((int)nEv_t, (int)nEv_s);
//      D.resize((int)nEv_t, (int)nEv_s);

      tdp::Timer t0;
      std::cout << "Calculating Bases&evals---" << std::endl;
      tdp::decomposeLaplacian(L_s, nEv_s, S_evals, S_wl); //todo: check if size initialization is included
      tdp::decomposeLaplacian(L_t, nEv_t, T_evals, T_wl);
      t0.toctic("Laplacian decomposition");

      tdp::gramSchmidt(S_wl);
      tdp::gramSchmidt(T_wl);
      t0.toctic("gramSchmidt");

//      //check if evectors are orthonormal
//      std::cout << "\n\nnormal LB basis check-----" << std::endl;
//      for (int i=0; i<S_wl.cols(); ++i){
//          std::cout << S_wl.col(i).norm() << ", ";
//      }
//      std::cout << "\nChecking if orthogonal------" << std::endl;
//      for (int i=0; i<S_wl.cols();++i ){
//        std::cout << "\nnew vec-----" << std::endl;
//        for (int j=0; j<i; ++j ){
//          std::cout << S_wl.col(i).dot(S_wl.col(j)) << ", ";
//        }
//      }
//     //end of print

      evector_s = S_wl.col(1); // first non-trivial evector
      evector_t = T_wl.col(1); // first non-trivial evector

      valuebo_s.Reinitialise(pangolin::GlArrayBuffer, evector_s.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_s.Upload(&evector_s(0), sizeof(float)*evector_s.rows(), 0);
      std::cout << evector_s.minCoeff() << " " << evector_s.maxCoeff() << std::endl;
      minVal_s = evector_s.minCoeff()-1e-3;
      maxVal_s = evector_s.maxCoeff();

      valuebo_t.Reinitialise(pangolin::GlArrayBuffer, evector_t.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_t.Upload(&evector_t(0), sizeof(float)*evector_t.rows(), 0);
      std::cout << evector_t.minCoeff() << " " << evector_t.maxCoeff() << std::endl;
      minVal_t = evector_t.minCoeff()-1e-3;
      maxVal_t = evector_t.maxCoeff();

      basisChanged = true;
      std::cout << "basisChanged" << std::endl;
    }

    if (pangolin::Pushed(showFMap) || nPW.GuiChanged() ||
        alpha2.GuiChanged() || basisChanged ){

      //--Construct function pairs
      Eigen::MatrixXf F((int)nPW, (int)nEv_s), G((int)nPW, (int)nEv_t),
                      C((int)nEv_s, (int)nEv_t), D((int)nEv_s, (int)nEv_s);

      std::cout << "nPW: " << (int)nPW << std::endl;
      std::cout << "F,G,C CREATED!---" << std::endl;
      std::cout << F.rows() << ", " << F.cols() << std::endl;
      std::cout << G.rows() << ", " << G.cols() << std::endl;
      std::cout << C.rows() << ", " << C.cols() << std::endl;

      std::cout << "S_WL---" << std::endl;
      std::cout << S_wl.rows() << ", " << S_wl.cols() << std::endl;
      std::cout << T_wl.rows() << ", " << T_wl.cols() << std::endl;

      // --construct F(data matrix) and G based on the correspondences
      for (int i=0; i< (int)nPW; ++i){
          Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area()), f_l, g_l;
//          tdp::f_landmark(pc_s, i, alpha2, option, f_w); //points in order in pc
//          tdp::f_landmark(pc_t, i, alpha2, option, g_w);
//          std::cout << "i: " << i << ", " << std::endl;
//          std::cout << "pindex: " << pIndices[i] << std::endl;

          tdp::f_landmark(pc_s, pIndices[i], alpha2, option, f_w); //points in suffled order
          tdp::f_landmark(pc_t, pIndices[i], alpha2, option, g_w);

          f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
          g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);

          F.row(i) = f_l.transpose();
          G.row(i) = g_l.transpose();

          if (i==0){
              std::cout << "f_l size: " << f_l.rows() << ", " << f_l.cols() << std::endl;
          }
      }

      // solve least-square
      C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);

//      std::cout << "-----------\n"
//                << "C(10x10) \n"
//                << C.block(0,0,10,10)
//                << std::endl;

      std::cout << "----------\n"
                << "Diagnoals\n"
                << C.diagonal().transpose()
                << std::endl;

      //Visualization of C
      tdp::GetGrid(pc_grid, (int)nEv_t, (int)nEv_s); //todo: order?
      vbo_cmtx.Reinitialise(pangolin::GlArrayBuffer, pc_grid.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_cmtx.Upload(pc_grid.ptr_, pc_grid.SizeBytes(), 0);

      //color coding of the C matrix
      Eigen::VectorXf cvec((int)nEv_t*(int)nEv_s);
      for (int r=0; r<C.rows(); ++r){
          for (int c=0; c<C.cols(); ++c){
              cvec(r*C.cols()+c) = C(r,c);
          }
      }

      valuebo_cmtx.Reinitialise(pangolin::GlArrayBuffer, cvec.rows(), GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_cmtx.Upload(&cvec(0), sizeof(float)*cvec.rows(), 0);

      std::cout << "<---CMTX--->" << std::endl;
      std::cout << cvec.minCoeff() << " " << cvec.maxCoeff() << std::endl;
      minVal_c = cvec.minCoeff()-1e-3;
      maxVal_c = cvec.maxCoeff();

      cChanged = true;
      std::cout << "C matrix is (re)calculated\n" << std::endl;

      // SVD for fmap semantic viz
      tdp::Timer t0;
      Eigen::JacobiSVD<Eigen::MatrixXf> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
      t0.toctic("SVD");
      std::cout << "C size: " << C.rows() << ", " << C.cols() << std::endl;
      std::cout << "Its singular values are:" << std::endl << svd.singularValues().transpose() << std::endl;
      std::cout << "U matrix (nEv_t):" << std::endl << svd.matrixU().rows() << ", " << (int)nEv_t << std::endl;
      std::cout << "V matrix (nEv_s):" << std::endl << svd.matrixV().rows() << ", " << (int)nEv_s <<std::endl;

      // calculate DiffMap
      D = C.transpose()*C;
      t0.toctic("shapeDiff");

      // For functionTransfer
      Eigen::VectorXf f_w(pc_s.Area());
      tdp::f_height(pc_s, f_w);
      Eigen::VectorXf f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
      Eigen::VectorXf g_l = (f_l.transpose() * C).transpose();
      Eigen::VectorXf g_w = T_wl * g_l;
      Eigen::VectorXf rawDiff = (g_w - f_w).array().square();

      // Viz of fMap
      //todo: if r.GuiChanged()
      Eigen::VectorXf v_r = svd.matrixV().col((int)r).array().square();
      Eigen::VectorXf u_r = svd.matrixU().col((int)r).array().square();
      f_w = S_wl * (v_r);
      g_w = T_wl * (u_r);

      std::cout << "v_r: " << v_r.transpose() << std::endl;
      std::cout << "u_r: " << u_r.transpose() << std::endl;




//      //For shapeDifference
//      //Take a function
//      Eigen::VectorXf f_w(pc_s.Area());
//      //sort dPoints
//      //get indicator function on the support
//      //tdp::f_indicator(pc_s, dPoints, f);
//      tdp::f_indicator(pc_s, dIndices, f_w);
//      std::cout << "checking f indicator f_w" << std::endl;
//      for (auto i = dIndices.begin(); i != dIndices.end(); ++i){
//        std::cout << f_w(*i) << " ";
//      }
//      std::cout << std::endl;
//      if ( dIndices.size() <= 0){
//        f_w.fill(1);
//      }
//      //To local
//      Eigen::VectorXf f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
//      Eigen::VectorXf d_f_l = D * f_l;
//      Eigen::VectorXf d_f_w = S_wl * d_f_l;
//      Eigen::VectorXf diffVec = (d_f_w - f_w).array().abs();
//      t0.toctic("diff operation");

      //Visualize fz and diff
      valuebo_f.Reinitialise(pangolin::GlArrayBuffer, f_w.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_f.Upload(&f_w(0), sizeof(float)*f_w.rows(), 0);
      valuebo_g.Reinitialise(pangolin::GlArrayBuffer, g_w.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_g.Upload(&g_w(0), sizeof(float)*g_w.rows(), 0);
      valuebo_d.Reinitialise(pangolin::GlArrayBuffer, rawDiff.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_d.Upload(&rawDiff(0), sizeof(float)*rawDiff.rows(), 0);
//      valuebo_d.Reinitialise(pangolin::GlArrayBuffer, d_f_w.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
//      valuebo_d.Upload(&d_f_w(0), sizeof(float)*d_f_w.rows(), 0);

      minVal_f = f_w.minCoeff()-1e-3;
      maxVal_f = f_w.maxCoeff()+1e-3;
      minVal_g = g_w.minCoeff()-1e-3;
      maxVal_g = g_w.maxCoeff()+1e-3;
      minVal_d = rawDiff.minCoeff()-1e-3;
      maxVal_d = rawDiff.maxCoeff()+1e-3;
//      minVal_d = d_f_w.minCoeff()-1e-3;
//      maxVal_d = d_f_w.maxCoeff()+1e-3;

      std::cout << "min max values------------->" << std::endl;
      std::cout << "cmtx (fmap): " << minVal_c << ", " << maxVal_c << std::endl;
      std::cout << "f_w: " << minVal_f << ", " << maxVal_f << std::endl;
      std::cout << "rawDiff: " << minVal_d << ", " << maxVal_d << std::endl;
//      std::cout << "d_fw-fw: " << minVal_d << ", " << maxVal_d << std::endl;
//      std::cout << "min, max d: " << minVal_d << ", " << maxVal_d << std::endl;

      std::cout << "--------FINAL RESULT---------" << std::endl;
      std::cout << "rawDiff norm: " << rawDiff.norm()/rawDiff.rows() << std::endl;
//      std::cout << "area ratio: " << (f_w.squaredNorm()/d_f_w.dot(f_w)) << std::endl;
    }


    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (view_s.IsShown()) {
      view_s.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minVal_s);
      shader.SetUniform("maxValue", maxVal_s);
      valuebo_s.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
      vbo_s.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo_s.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      valuebo_s.Unbind();
      glDisableVertexAttribArray(0);
      vbo_s.Unbind();
    }

    if (view_t.IsShown()){
        view_t.Activate(s_cam);
        pangolin::glDrawAxis(0.1);

        glPointSize(2.);
        glColor3f(1.0f, 1.0f, 0.0f);
        // renders the vbo with colors from valuebo
        auto& shader_t = tdp::Shaders::Instance()->valueShader_;
        shader_t.Bind();
        shader_t.SetUniform("P",  s_cam.GetProjectionMatrix());
        shader_t.SetUniform("MV", s_cam.GetModelViewMatrix());
        shader_t.SetUniform("minValue", minVal_t);
        shader_t.SetUniform("maxValue", maxVal_t);
        valuebo_t.Bind();
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
        vbo_t.Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(4.);
        glDrawArrays(GL_POINTS, 0, vbo_t.num_elements);
        shader_t.Unbind();
        glDisableVertexAttribArray(1);
        valuebo_t.Unbind();
        glDisableVertexAttribArray(0);
        vbo_t.Unbind();
    }

    if (view_cmtx.IsShown()){
        view_cmtx.Activate(cmtx_cam);
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

    if (view_f.IsShown()) {
      view_f.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minVal_f);
      shader.SetUniform("maxValue", maxVal_f);
      valuebo_f.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
      vbo_s.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo_s.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      valuebo_f.Unbind();
      glDisableVertexAttribArray(0);
      vbo_s.Unbind();
    }

    if (view_g.IsShown()) {
      view_g.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minVal_g);
      shader.SetUniform("maxValue", maxVal_g);
      valuebo_g.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
      vbo_t.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo_s.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      valuebo_g.Unbind();
      glDisableVertexAttribArray(0);
      vbo_t.Unbind();
    }


    if (view_d.IsShown()) {
      view_d.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minVal_d);
      shader.SetUniform("maxValue", maxVal_d);
      valuebo_d.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
      vbo_s.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo_s.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      valuebo_d.Unbind();
      glDisableVertexAttribArray(0);
      vbo_s.Unbind();
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

  std::cout << "AY YO!" << std::endl;
  return 0;
}

