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
#include <tdp/manifold/SE3.h>

#include <tdp/utils/status.h>

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <Eigen/Eigenvalues>
#include "laplace_beltrami.h"

float f_z(const tdp::Vector3fda& x) {
    return x(2);
}

tdp::Vector3fda getMean(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  assert(pc.rows() == 1);
  tdp::Vector3fda mean(0,0,0);
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

inline void getAxesIds(const std::vector<auto>& vec, std::vector<int>& sortIds){
    int hi(0), lo(0), mid;
    for (size_t i=0; i<vec.size(); ++i){
        if (vec[i] < vec[lo]){
            lo = i;
        }else if (vec[i] > vec[hi]){
            hi = i;
        }
    }

    for (size_t i=0; i<vec.size();++i){
        if (i!=hi&&i!=lo){
            mid=i;
        }
    }
    sortIds.push_back(hi);
    sortIds.push_back(mid);
    sortIds.push_back(lo);
}

Eigen::Matrix3f getLocalRot(const tdp::Matrix3fda& cov, const Eigen::SelfAdjointEigenSolver<tdp::Matrix3fda>& es){
    std::vector<float> evalues;
    std::vector<int> axesIds;
    for (size_t i=0; i<cov.rows(); ++i){
        float eval = std::real(es.eigenvalues().col(0)[i]);
        evalues.push_back( (eval<1e-6? 0: eval));
    }

    getAxesIds(evalues,axesIds);

    Eigen::Matrix3f localRot;
    for (size_t i=0; i<3; ++i){
        localRot.col(i) = es.eigenvectors().col(axesIds[i]);
    }
    return localRot;
}


void getAllLocalBasis(const tdp::Image<tdp::Vector3fda>& pc, tdp::Image<tdp::SE3f>& T_wl, tdp::ANN& ann, int knn, float eps){
    //assumes ANN has complete computing kd tree
    //query `knn` number of neighbors
    assert( (pc.w_==T_wl.w_)&&(pc.h_ == T_wl.h_) );

    tdp::Vector3fda query, localMean;
    tdp::Matrix3fda cov, localRot;
    Eigen::SelfAdjointEigenSolver<tdp::Matrix3fda> es;
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    for (size_t i = 0; i<pc.Area(); ++i){
        query = pc(i,0);
        ann.Search(query, knn, eps, nnIds, dists);
        cov = getCovariance(pc,nnIds);
        es.compute(cov);
        localRot = getLocalRot(cov,es);
        localMean = getMean(pc, nnIds);
        T_wl[i] = tdp::SE3f(localRot, localMean);
        tdp::Progress(i,pc.Area());
    }
}

inline float w(float d, int knn){
    return d==0? 1: 1/(float)knn;
}

inline tdp::Vector6fda poly2Basis(const tdp::Vector2fda& vec){
    tdp::Vector6fda newVec;
    newVec << 1, vec[0], vec[1], vec[0]*vec[0], vec[0]*vec[1], vec[1]*vec[1];
    return newVec;
}

inline Eigen::Vector4f homogeneous(const tdp::Vector3fda& p){
    return tdp::Vector4fda(p(0),p(1),p(2),1);
}

void getThetas(const tdp::Image<tdp::Vector3fda>& pc_w, const tdp::Image<tdp::SE3f>& T_wls, tdp::Image<tdp::Vector6fda>& thetas, tdp::ANN& ann, int knn, float eps){
    assert(pc_w.w_ == T_wls.w_&&pc_w.w_==thetas.w_);
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    for (size_t i=0; i<pc_w.Area(); ++i){
        tdp::Vector3fda pt = pc_w[i];
        const tdp::SE3f& T_wl = T_wls[i];

        // Get the neighbor ids and dists for this point
        ann.Search(pt, knn, eps, nnIds, dists);

        tdp::MatrixXfda X(knn,6), W(knn,knn);//todo clean this up
        tdp::VectorXfda Y(knn);
        tdp::Vector6fda theta;
        for (size_t k=0; k<knn; ++k){
            //std::cout << "iter: " << k << std::endl;
            //std::cout << "kth neighbor pt in wc: \n" << pc(nnIds[k],0) <<std::endl;
            tdp::Vector3fda npt_l_ = T_wl.Inverse()*pc_w[nnIds[k]];
            //Take the first two dimensions
            tdp::Vector2fda npt_2d(npt_l_(0), npt_l_(1));
            //target is the third dim coordinate
            float npt_z = npt_l_(2);
            //project to higher dimension using poly2 basis
            tdp::Vector6fda phi_npt = poly2Basis(npt_2d);
            //Construct data matrix X
            X.row(k) = phi_npt;
            //Construct target vector Y
            Y(k) = npt_z;
            //Get weight matrix W
            W(k,k) = dists(k); //check if I need to scale this when in local coordinate system
        }

        //Solve weighted least square
        Eigen::FullPivLU<tdp::Matrix6fda> X_lu;
        X_lu.compute(X.transpose()*W*X);
        theta = X_lu.solve(X.transpose()*W*Y);
        thetas[i] = theta;
    }
}
void getZEstimates(const tdp::Image<tdp::Vector3fda>& pc_w,
                   const tdp::Image<tdp::SE3f>& T_wl,
                   const tdp::Image<tdp::Vector6fda>& thetas,
                   tdp::Image<tdp::Vector3fda>& estimates_w){
    tdp::Vector3fda pt_l, wc_z_estimated;
    tdp::Vector6fda phi_pt, theta;
    float z_estimated;
    for (size_t i=0; i<pc_w.Area(); ++i){
        pt_l = T_wl[i].Inverse()*pc_w[i];
        theta = thetas[i];
        //Estimate normals
        phi_pt = poly2Basis(tdp::Vector2fda(pt_l(0), pt_l(1)));
        z_estimated = theta.transpose()*phi_pt;\
        estimates_w[i] = T_wl[i]*(tdp::Vector3fda(pt_l(0),pt_l(1),z_estimated));

//        wc_z_estimated = locals[i].Inverse()*(tdp::Vector3fda(pt(0),pt(1),z_estimated));
//        estimates(i,0) = tdp::Vector3fda(pt(0), pt(1), wc_z_estimated(2));
   }
}

void getSamples(const tdp::Image<tdp::SE3f>& T_wl,
                const tdp::Image<tdp::Vector6fda>& thetas,
                tdp::Image<tdp::Vector3fda>& estimates_w, size_t upsample){
    tdp::Vector3fda pt_l, wc_z_estimated;
    tdp::Vector6fda phi_pt, theta;
    float z_estimated;
    for (size_t i=0; i<T_wl.Area(); ++i){
        for (size_t j=0; j<upsample; ++j) {
            pt_l = 0.1*tdp::Vector3fda::Random();
            //Estimate normals
            phi_pt = poly2Basis(tdp::Vector2fda(pt_l(0), pt_l(1)));
            z_estimated = thetas[i].transpose()*phi_pt;\
            estimates_w[i*upsample+j] = T_wl[i]*(tdp::Vector3fda(pt_l(0),pt_l(1),z_estimated));
        }
//        wc_z_estimated = locals[i].Inverse()*(tdp::Vector3fda(pt(0),pt(1),z_estimated));
//        estimates(i,0) = tdp::Vector3fda(pt(0), pt(1), wc_z_estimated(2));
   }
}


//tests
void test_meanAndCov(){
        //TEST OF getMean and getCovariance
        tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();
        Eigen::VectorXi nnIds(10);
        nnIds<< 0,1,2,3,4,5,6,7,8,9;
        tdp::Vector3fda mean = getMean(pc, nnIds);
        tdp::Matrix3fda cov = getCovariance(pc,nnIds);
        std::cout << "mean: \n" << mean << std::endl << std::endl;
        std::cout << "cov: \n" << cov << std::endl << std::endl;
}

void test_getAllLocalBasis(){
    //test getAllLocalBasis
    tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();
    tdp::ManagedHostImage<tdp::SE3f> locals(pc.w_,1);

    tdp::ANN ann;
    ann.ComputeKDtree(pc);

    int knn = 5;
    float eps = 1e-4;
    getAllLocalBasis(pc, locals, ann,knn, eps);

    for (size_t i=0; i<1/*locals.Area()*/; ++i){
        std::cout << "point: \n " << pc(i,0) << std::endl;
        std::cout << "localbasis: \n"<<locals(i,0) << std::endl << std::endl;
    }
}

void test_getAxesIds(){
    std::vector<int> v = {1,5,3};
    std::vector<int> ids;
    getAxesIds(v,ids);
    for (int i =0; i<ids.size(); ++i){
        std::cout << ids[i] << ": "<< v[ids[i]] << std::endl;
    }
}

void test_poly2Basis(){
    tdp::Vector2fda vec1(10.,10.);
    tdp::Vector2fda vec2(0,0);
    std::cout << poly2Basis(vec1) << std::endl;
    std::cout << poly2Basis(vec2) << std::endl;

}

void test_getLocalRot(){
    tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();
    tdp::Vector3fda query;
    tdp::Matrix3fda cov, localRot;
    Eigen::SelfAdjointEigenSolver<tdp::Matrix3fda> es;
    int knn = 1;
    float eps = 1e-4;
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);

    tdp::ANN ann;
    ann.ComputeKDtree(pc);

    for (size_t i = 0; i<pc.Area(); ++i){
        query = pc(i,0);
        ann.Search(query, knn, eps, nnIds, dists);
        cov = getCovariance(pc,nnIds);
        es.compute(cov);
        localRot = getLocalRot(cov,es);

        std::cout << "\niter: " << i << std::endl;
        std::cout << "curr pt: \n" << query << std::endl;
        //std::cout << "neighbors: \n" << nnIds << std::endl;
        std::cout << "cov: \n" << cov << std::endl;
        std::cout << "localRot: \n" << localRot << std::endl;
        std::cout << "\t result: \n" << localRot*query << std::endl;
    }
}

//end of test

int main( int argc, char* argv[] ){
    //test_getLocalRot();
    //return 1;
    //test1();
    //return 1;
  // load pc and normal from the input paths
  //tdp::ManagedHostImage<tdp::Vector3fda> pc=GetSimplePc();
  tdp::ManagedHostImage<tdp::Vector3fda> pc(10000,1);

  tdp::ManagedHostImage<tdp::Vector3fda> ns(10000,1);

  if (argc > 1) {
      const std::string input = std::string(argv[1]);
      std::cout << "input " << input << std::endl;
      tdp::LoadPointCloud(input, pc, ns);
  } else {
      GetSphericalPc(pc);
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
  pangolin::GlBuffer vbo, vboM, vboS;
  vbo.Reinitialise(pangolin::GlArrayBuffer, pc.Area(),  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  vbo.Upload(pc.ptr_, pc.SizeBytes(), 0);

  // Add variables to pangolin GUI
  pangolin::Var<bool> runSkinning("ui.run skinning", true, false);
  pangolin::Var<bool> showBases("ui.show bases", true, true);

  pangolin::Var<int> knn("ui.knn", 10,1,100);

  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);
  pangolin::Var<int> upsample("ui.upsample", 10,1,100);


  tdp::ManagedHostImage<tdp::SE3f> T_wl(pc.w_,1);
  tdp::ManagedHostImage<tdp::Vector6fda> thetas(pc.w_,1);
  tdp::ManagedHostImage<tdp::Vector3fda> zEstimates(pc.w_,1);
  tdp::ManagedHostImage<tdp::Vector3fda> zSamples(pc.w_*upsample,1);





  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (pangolin::Pushed(runSkinning) || knn.GuiChanged() || upsample.GuiChanged()) {
        //  processing of PC for skinning
      std::cout << "Running skinning..." << std::endl;

      getAllLocalBasis(pc, T_wl, ann, knn, eps);

      getThetas(pc,T_wl,thetas,ann,knn,eps);

      getZEstimates(pc,T_wl,thetas,zEstimates);
      vboM.Reinitialise(pangolin::GlArrayBuffer, zEstimates.Area() , GL_FLOAT, 3, GL_DYNAMIC_DRAW ); //will later be knn*pc.Area()
      vboM.Upload(zEstimates.ptr_, sizeof(tdp::Vector3fda) * zEstimates.Area(), 0);

      zSamples.Reinitialise(pc.w_*upsample,1);
      getSamples(T_wl,thetas,zSamples,upsample);

      // put the estimated points to GLBuffer vboM
      vboS.Reinitialise(pangolin::GlArrayBuffer, zSamples.Area() , GL_FLOAT, 3, GL_DYNAMIC_DRAW ); //will later be knn*pc.Area()
      vboS.Upload(zSamples.ptr_, sizeof(tdp::Vector3fda) * zSamples.Area(), 0);

      std::cout << "<--DONE skinning-->" << std::endl;
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);
    if (viewPc.IsShown()) {
      viewPc.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      if (showBases) {
          for (size_t i=0; i<T_wl.Area(); ++i) {
              pangolin::glDrawAxis(T_wl[i].matrix(), 0.05f);
          }
      }

      glPointSize(2.);
      glColor3f(1.0f, 0.0f, 1.0f);
      pangolin::RenderVbo(vboM);

      glPointSize(1.);
      glColor3f(0.0f, 1.0f, 1.0f);
      pangolin::RenderVbo(vboS);

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

  std::cout << "good morning!" << std::endl;
  return 0;
}
