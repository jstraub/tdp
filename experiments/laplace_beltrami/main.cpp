/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>


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

int main( int argc, char* argv[] ){

    std::srand(101);
    //test_Laplacian();
   //return 1;
  // load pc and normal from the input paths
  tdp::ManagedHostImage<tdp::Vector3fda> pc(10000,1);
  tdp::ManagedHostImage<tdp::Vector3fda> ns(10000,1);

  if (argc > 1) {
      const std::string input = std::string(argv[1]);
      std::cout << "input pc: " << input << std::endl;
      tdp::LoadPointCloud(input, pc, ns);
  } else {
      GetSphericalPc(pc);
      //GetCylindricalPc(pc);
  }

  // build kd tree
  tdp::ANN ann;
  ann.ComputeKDtree(pc);
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
  pangolin::GlBuffer vbo, vboM, vboS, vboF, valuebo;
  vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);

  // Add variables to pangolin GUI
  pangolin::Var<bool> runSkinning("ui.run skinning", true, false);
  pangolin::Var<int> pcOption("ui. pc option", 0, 0,1);
  pangolin::Var<bool> showBases("ui.show bases", true, true);
  // variables for KNN
  pangolin::Var<int> knn("ui.knn",30,1,100);
  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);
  pangolin::Var<int> numEv(" ui. num_EV", 10,10,100);
  pangolin::Var<int> idEv("ui.id EV", 1, 0, numEv-1);
  pangolin::Var<float> alpha("ui. alpha", 0.01, 0.005, 0.3); //variance of rbf kernel
  // viz color coding
  pangolin::Var<float>minVal("ui. min Val",-0.71,-1,0);
  pangolin::Var<float>maxVal("ui. max Val",0.01,1,0);
  pangolin::Var<bool> recomputeEvector("ui.compute evector", true, false);

  pangolin::Var<bool> recomputeMeans("ui.compute means", true, false);
  pangolin::Var<int>nBins("ui. nBins", 10, 10,100);
  // sampling
  pangolin::Var<int> upsample("ui.upsample", 10,1,100);

  tdp::ManagedHostImage<tdp::SE3f> T_wls(pc.w_,1);
  //tdp::ManagedHostImage<tdp::Vector6fda> thetas(pc.w_,1);
  //tdp::ManagedHostImage<tdp::Vector3fda> zEstimates(pc.w_,1),fEstimates(pc.w_,1);
  //tdp::ManagedHostImage<tdp::Vector3fda> zSamples(pc.w_*upsample,1);
  Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());
  tdp::eigen_vector<Eigen::VectorXf> evectors(numEv, Eigen::VectorXf(pc.Area(),1));
  Eigen::VectorXf evector(pc.Area(),1);
  Eigen::MatrixXf curvature(pc.Area(),3);

  tdp::eigen_vector<tdp::Vector3fda> means;
  means.reserve(nBins);
  // Stream and display video
  while(!pangolin::ShouldQuit()){
      // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if( pcOption.GuiChanged()){
      std::cout << "Roading new pc..." << pcOption << std::endl;
      if (pcOption == 0){
          GetSphericalPc(pc);
      } else if (pcOption ==1 ){
          GetCylindricalPc(pc);
      }
    }

    if (pangolin::Pushed(runSkinning) || knn.GuiChanged() ||
            upsample.GuiChanged()  || alpha.GuiChanged() ||
            numEv.GuiChanged()) {
        //  processing of PC for skinning
      std::cout << "Running skinning..." << std::endl;

     // getAllLocalBasis(pc, T_wls, ann, knn, eps);
//      getThetas(pc,T_wls,thetas,ann,knn,eps);
//      getZEstimates(pc,T_wls,thetas,zEstimates);
//      vboZ.Reinitialise(pangolin::GlArrayBuffer, zEstimates.Area() , GL_FLOAT, 3, GL_DYNAMIC_DRAW ); //will later be knn*pc.Area()
//      vboZ.Upload(zEstimates.ptr_, sizeof(tdp::Vector3fda) * zEstimates.Area(), 0);

      //getThetas_F(pc, T_wls, f_z, thetas, ann, knn, eps);
      //getFEstimates(pc, T_wls, thetas, fEstimates);
      //vboF.Reinitialise(pangolin::GlArrayBuffer, fEstimates.Area() , GL_FLOAT, 3, GL_DYNAMIC_DRAW ); //will later be knn*pc.Area()
      //vboF.Upload(fEstimates.ptr_, sizeof(tdp::Vector3fda) * fEstimates.Area(), 0);

      //zSamples.Reinitialise(pc.w_*upsample,1);
      //getSamples(T_wls,thetas,zSamples,upsample);

      // get Laplacian operator and its eigenvectors
      std::cout << "Calculating L, and a bunch of evectors..." << std::endl;
      tdp::Timer t0;
      L = tdp::getLaplacian(pc, ann, knn, eps, alpha);
      t0.toctic("GetLaplacian");
      //evector = tdp::getLagetLaplacianEvector(pc, L, idEv);
      tdp::getLaplacianEvectors(L, numEv, evectors);
      t0.toctic("Get bunch of evectors");
      recomputeEvector = true;
    }

    if (pangolin::Pushed(recomputeEvector) || idEv.GuiChanged()){
        evector = evectors[idEv];
        std::cout << "new idEv. simple selection." << std::endl;

        // color-coding on the surface
        valuebo.Reinitialise(pangolin::GlArrayBuffer, evector.rows(),
                             GL_FLOAT,1, GL_DYNAMIC_DRAW);
        valuebo.Upload(&evector(0), sizeof(float)*evector.rows(), 0);
        std::cout << evector.minCoeff() << " " << evector.maxCoeff() << std::endl;
        minVal = evector.minCoeff()-1e-3;
        maxVal = evector.maxCoeff();
        recomputeMeans = true;

    }

    if (pangolin::Pushed(recomputeMeans) || nBins.GuiChanged()) {
        std::cout << "recomputeMean" << std::endl;
        means = tdp::getLevelSetMeans(pc, evector, nBins);
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);
    if (viewPc.IsShown()) {
      viewPc.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      if (showBases) {
          for (size_t i=0; i<T_wls.Area(); ++i) {
              pangolin::glDrawAxis(T_wls[i].matrix(), 0.05f);
          }
      }

      glPointSize(2.);
      glColor3f(1.0f, 1.0f, 0.0f);
      pangolin::RenderVbo(vboF);

      //glPointSize(10.);
      //glColor3f(.5f, 1.f, .5f);
      //pangolin::RenderVbo(vboM);

      glColor3f(.3,1.,.125);
      glLineWidth(2);
      tdp::Vector3fda m, m_prev;
      for (size_t i=1; i<means.size(); ++i){
          m_prev = means[i-1];
          m = means[i];
          tdp::glDrawLine(m_prev, m);
      }

      //glPointSize(1.);
      // draw the first arm pc
      //glColor3f(1.0f, 0.0f, 0.0f);
      //pangolin::RenderVbo(vbo);


      // renders the vbo with colors from valuebo
      auto& shader = tdp::Shaders::Instance()->valueShader_;   
      shader.Bind();
      shader.SetUniform("P",  s_cam.GetProjectionMatrix());
      shader.SetUniform("MV", s_cam.GetModelViewMatrix());
      shader.SetUniform("minValue", minVal);
      shader.SetUniform("maxValue", maxVal);
      valuebo.Bind();
      glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0); 
      vbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 

      glEnableVertexAttribArray(0); 
      glEnableVertexAttribArray(1);
      glPointSize(4.);
      glDrawArrays(GL_POINTS, 0, vbo.num_elements);
      shader.Unbind();
      glDisableVertexAttribArray(1);
      valuebo.Unbind();
      glDisableVertexAttribArray(0);
      vbo.Unbind();


    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();

    //todo?
    recomputeMeans = false;
    recomputeEvector = false;
  }

  std::cout << "good morning!" << std::endl;
  return 0;
}
