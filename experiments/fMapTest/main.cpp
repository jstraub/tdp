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
std::map<std::string, std::string> makeCacheNames(
        const int shapeOpt, 
        const int nSamples, 
        const int knn, 
        const float alpha,
        const int nEv, 
        const char* cacheDir="./cache/");

void Test_makeCacheNames();


int main(int argc, char* argv[]){
  Test_makeCacheNames();
  return 0;
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
                          .SetHandler(new pangolin::Handler3D(t_cam));
  pangolin::View& view_cmtx = pangolin::CreateDisplay()
                          .SetHandler(new pangolin::Handler3D(cmtx_cam));
  container.AddDisplay(view_s);
  container.AddDisplay(view_t);
  container.AddDisplay(view_cmtx);
    
  // Add variables to pangolin GUI
  pangolin::Var<bool> showFMap("ui.show fMap", true, false);
  pangolin::Var<bool> showMeans("ui.show means", true, false);
  pangolin::Var<bool> showFTransfer(" ui. show fTransfer", true, true);
  pangolin::Var<bool> showDecomposition("ui. show evals", false, true);

  pangolin::Var<int> nSamples("ui. num samples from mesh pc", 400, 100, 400);

  pangolin::Var<int> shapeOpt("ui. shape option", 2, 0, 3); //2:bunny
  //-- variables for KNN
  pangolin::Var<int> knn("ui.knn",30,1,100);//(int)nSamples
  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);
  pangolin::Var<float> alpha("ui. alpha", 0.01, 0.001, 0.3); //variance of rbf kernel for laplacian
  pangolin::Var<float> alpha2("ui. alpha2", 0.001, 0.001, 0.5); //variance of rbf kernel for defining function on manifold
  pangolin::Var<int> nBins("ui. nBins", 10, 10,100);

  //--Correspondence Matrix C estimation
  pangolin::Var<int> nEv("ui.numEv",50,30,100); //min=1, max=pc_s.Area()
  pangolin::Var<int> nPW("ui.numCst",nEv/*std::min(20*numEv, pc_s.Area())*/,nEv,nSamples);

  pangolin::Var<int> nCst("ui.numCst",nPW,nPW,nSamples);
  pangolin::Var<int> nSteps("ui.nSteps for HKS", 100, 50,300); //number of timesteps for HKS
  //-- viz color coding
  pangolin::Var<float>minVal("ui. min Val",-0.71,-1,0);
  pangolin::Var<float>maxVal("ui. max Val",0.01,1,0);
  pangolin::Var<int>nQ("ui. num Queries", 100, 100, nSamples);

  // use those OpenGL buffers
  pangolin::GlBuffer vbo_s, vbo_t, vbo_cmtx,
                     vbo_f, vbo_g,  //point clouds
                     vbo_f0, vbo_g0,

                     valuebo_s, valuebo_t, valuebo_cmtx, valuebo_color, //colorings: source manifold, target manifod, c_mtx
                     valuebo_f0, valuebo_g0,
                     valuebo_desc_s, valuebo_desc_t; //shape descriptor values (hks,wks)

  // min,max for coloring
  float minVal_t, maxVal_t, minVal_c, maxVal_c, minCValue, maxCValue,
        minF0Value, maxF0Value, minG0Value, maxG0Value;

  // Control switches
  bool annChanged = false;
  bool laplacianChanged = false;
  bool basisChanged = false;
  bool cChanged = false;
  bool queryChanged = false;

  /******************End of Pangolin GUI setup*************************/

  //*****************Declare variables*********************************/
  std::string option("rbf");

  tdp::ManagedHostImage<tdp::Vector3fda> pc_all;
  tdp::ManagedHostImage<tdp::Vector3fda> pc_s;
  tdp::ManagedHostImage<tdp::Vector3fda> pc_t;
  tdp::ManagedHostImage<tdp::Vector3fda> pc_grid((int)nEv*(int)nEv,1);

  std::string fpath_b, fpath_m;
  std::map<std::string, std::string> cacheDic;
  tdp::ANN ann_s, ann_t;


  Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area()),//Laplacian of manifold S
                             L_t(pc_t.Area(), pc_t.Area());//Laplacian of manifold T
  Eigen::VectorXf evector_s,//(L_s.rows());
                  evector_t,//(L_t.rows());
                  S_evals,//evalues of Laplacian of S. Increasing order.
                  T_evals;//evalues of Laplacian of T. Increasing order.

  Eigen::MatrixXf S_wl,//(L_s.rows(),(int)numEv),
                  T_wl,//(L_t.rows(),(int)numEv),
                  S_desc_w, //(L_s rows(), (int)nSteps
                  T_desc_w,
                  F,//((int)numCst, (int)numEv),
                  G,//((int)numCst, (int)numEv),
                  C;//((int)nEv, (int)nEv);
  /*********************************************************************/

  // Stream and display video
  while (!pangolin::ShouldQuit()){
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Get samples
    if ( pangolin::Pushed(showFMap) || 
         nSamples.GuiChanged()      ||
         shapeOpt.GuiChanged()      ){
      std::cout << "Running fMap from top..." << std::endl;

      if (argc<=3){
          std::cout << "Give paths to bunny and manekin ply files. Terminated." << std::endl;
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
            std::cout << "Shape: linear---" << std::endl;
            break;
          case 1:
            tdp::GetSphericalPc(pc_s, nSamples);
            std::cout << "Shape: sphere---" << std::endl;
            break;
          case 2:
            tdp::LoadPointCloudFromMesh(fpath_b, pc_all);
            tdp::GetSamples(pc_all, pc_s, nSamples);
            std::cout << "Shape: bunny---" << std::endl;
            break;
          case 3:
            tdp::ManagedHostImage<tdp::Vector3fda> pc_all;
            tdp::LoadPointCloudFromMesh(fpath_m, pc_all);
            tdp::GetSamples(pc_all, pc_s, nSamples);
            std::cout << "Shape: manekine---" << std::endl;
        }

        pc_t.ResizeCopyFrom(pc_s);

        std::cout << "PC_S: " << pc_s.Area() << std::endl;
        std::cout << "PC_T: " << pc_t.Area() << std::endl;

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

        //******************CACHE NAMING*************************//
        //*******************************************************//
        cacheDic = makeCacheNames((int)shapeOpt, (int)nSamples, (int)knn, 
                                  (float)alpha, (int)nEv);
        const char* path_ls = cacheDic.at("ls").c_str();
        const char* path_lt = cacheDic.at("lt").c_str();
        

        //***************Get Laplacians***************************//
        tdp::Timer t0;
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
        
            std::cout << "Cached Laplacians---" << std::endl;
        }

        t0.toctic("GetLaplacians");
        laplacianChanged = true;
        std::cout << "LaplacianS changed\n" << std::endl;
    }

    if (pangolin::Pushed(showFMap) || laplacianChanged ||
                nEv.GuiChanged() || nBins.GuiChanged()){
      S_wl.resize(L_s.rows(),(int)nEv);
      T_wl.resize(L_t.rows(),(int)nEv);
      S_evals.resize((int)nEv);
      T_evals.resize((int)nEv);
      S_desc_w.resize(L_s.rows(), (int)nSteps);
      T_desc_w.resize(L_t.rows(), (int)nSteps);

      //******************CACHE NAMING*************************//
      const char* path_s_wl = cacheDic.at("s_wl").c_str();
      const char* path_t_wl = cacheDic.at("t_wl").c_str();
      const char* path_s_evals = cacheDic.at("s_evals").c_str();
      const char* path_t_evals = cacheDic.at("t_evals").c_str();

      std::cout << "s_wl resized: " << S_wl.rows() << ", " << S_wl.cols() << std::endl;
      std::cout << "t_wl resized: " << T_wl.rows() << ", " << T_wl.cols() << std::endl;

      tdp::Timer t0;
      int res = access(path_s_wl, R_OK) + access(path_t_wl, R_OK) + 
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

      t0.toctic("GetEigenVectors & GetMeans");

      evector_s = S_wl.col(1); // first non-trivial evector
      evector_t = T_wl.col(1); // first non-trivial evector

      valuebo_s.Reinitialise(pangolin::GlArrayBuffer, evector_s.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_s.Upload(&evector_s(0), sizeof(float)*evector_s.rows(), 0);
      std::cout << evector_s.minCoeff() << " " << evector_s.maxCoeff() << std::endl;
      minVal = evector_s.minCoeff()-1e-3;
      maxVal = evector_s.maxCoeff();

      valuebo_t.Reinitialise(pangolin::GlArrayBuffer, evector_t.rows(),GL_FLOAT,1, GL_DYNAMIC_DRAW);
      valuebo_t.Upload(&evector_t(0), sizeof(float)*evector_t.rows(), 0);
      std::cout << evector_t.minCoeff() << " " << evector_t.maxCoeff() << std::endl;
      minVal_t = evector_t.minCoeff()-1e-3;
      maxVal_t = evector_t.maxCoeff();

      basisChanged = true;
      std::cout << "basisChanged" << std::endl;
    }

    if (pangolin::Pushed(showFMap) || 
        nCst.GuiChanged()        || 
        alpha2.GuiChanged()        ||
        basisChanged                ){

      F.resize((int)nCst, (int)nEv);
      G.resize((int)nCst, (int)nEv);
      C.resize((int)nEv, (int)nEv);
      Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area());
      Eigen::VectorXf f_l((int)nEv), g_l((int)nEv);

      // --construct F(data matrix) and G based on the correspondences
      for (int i=0; i< (int)nPW; ++i){
          tdp::f_landmark(pc_s, i, alpha2, option, f_w);
          tdp::f_landmark(pc_t, i, alpha2, option, g_w);

          f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
          g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);

          F.row(i) = f_l;
          G.row(i) = g_l;
      }

      // solve least-square
      C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);

      std::cout << "-----------\n"
                << "C(10x10) \n" 
                << C.block(0,0,10,10) 
                << std::endl;

      std::cout << "----------\n"
                << "Diagnoals\n"
                << C.diagonal().transpose() 
                << std::endl;

      //Visualization of C
      tdp::GetGrid(pc_grid, (int)nEv, (int)nEv);
      vbo_cmtx.Reinitialise(pangolin::GlArrayBuffer, pc_grid.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_cmtx.Upload(pc_grid.ptr_, pc_grid.SizeBytes(), 0);

      //color coding of the C matrix
      Eigen::VectorXf cvec((int)nEv*(int)nEv);
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
      std::cout << "C matrix is (re)calculated\n" << std::endl;
    }

    if (cChanged){
      vbo_f0.Reinitialise(pangolin::GlArrayBuffer, pc_s.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_s.Upload(pc_s.ptr_, pc_s.SizeBytes(), 0);
      vbo_t.Reinitialise(pangolin::GlArrayBuffer, pc_t.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
      vbo_t.Upload(pc_t.ptr_, pc_t.SizeBytes(), 0);
      std::cout << "<--DONE fMap-->" << std::endl;
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
      shader.SetUniform("minValue", minVal);
      shader.SetUniform("maxValue", maxVal);
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
        view_t.Activate(t_cam);
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


std::map<std::string, std::string> makeCacheNames(
        const int shapeOpt, 
        const int nSamples, 
        const int knn, 
        const float alpha,
        const int nEv, 
        const char* cacheDir){

  std::map<std::string, std::string> d;
  std::stringstream ss;

  ss << cacheDir << shapeOpt << "/ls_" << nSamples << "_" << knn << "_"
      << alpha << ".dat";
  d["ls"] = ss.str();
  ss.str(std::string());

  ss << cacheDir << shapeOpt << "/lt_" << nSamples << "_" << knn << "_"
      << alpha << ".dat";
  d["lt"] = ss.str();
  ss.str(std::string());

  ss << cacheDir << shapeOpt << "/s_wl_" << nSamples << "_" << nEv
     << ".dat";
  d["s_wl"] = ss.str();
  ss.str(std::string());

  ss << cacheDir << shapeOpt << "/t_wl_" << nSamples << "_" << nEv
     << ".dat";
  d["t_wl"] = ss.str();

  ss.str(std::string());

  ss << cacheDir << shapeOpt << "/s_evals_" << nSamples << "_" << nEv
     << ".dat";
  d["s_evals"] = ss.str();
  ss.str(std::string());

  ss << cacheDir << shapeOpt << "/t_evals_" << nSamples << "_" << nEv
     << ".dat";
  d["t_evals"] = ss.str();
  ss.str(std::string());

  return d;
}

void Test_makeCacheNames(){
  int shapeOpt = 0;
  int nSamples = 10;
  int knn = 10;
  float alpha = 0.01;
  int nEv = 5;

  std::map<std::string, std::string> d = makeCacheNames(
    shapeOpt, nSamples, knn, alpha, nEv, "./cache/");

  std::cout << "Checking the dictionary---" << std::endl;
  for (auto& k : d){
    std::cout << k.first << ", " << k.second << std::endl;
  }

  std::cout << "\nTEST2----" << std::endl;
  std::map<std::string, std::string> cacheDic = makeCacheNames(
    shapeOpt, nSamples, knn, alpha, nEv, "./somedir/");
  const char* path_ls = cacheDic.at("ls").c_str();
  const char* path_lt = cacheDic.at("lt").c_str();
  const char* path_s_wl = cacheDic.at("s_wl").c_str();
  const char* path_t_wl = cacheDic.at("t_wl").c_str();
  const char* path_s_evals = cacheDic.at("s_evals").c_str();
  const char* path_t_evals = cacheDic.at("t_evals").c_str();

  std::cout << "checking here!!--" << std::endl;
  std::cout << path_ls << std::endl;
  std::cout << path_lt << std::endl;
  std::cout << path_s_wl << std::endl;
  std::cout << path_t_wl << std::endl;
  std::cout << path_s_evals << std::endl;
  std::cout << path_t_evals << std::endl;
}

