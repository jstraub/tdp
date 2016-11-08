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
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>


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
#include <tdp/data/managed_image.h>


#include <tdp/utils/status.h>

#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>
#include "laplace_beltrami.h"

#include <tdp/gl/shaders.h>

float f_z(const tdp::Vector3fda& x) {
    return x(2);
}
\
float f_etoz(const tdp::Vector3fda& x){
//    return (float)exp(x(2));
    return x(2)*x(2);
}

tdp::Vector3fda getMean(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  assert(pc.h_ == 1);
  tdp::Vector3fda mean(0,0,0);
  for (size_t i=0; i<nnIds.rows(); ++i){
      mean +=  pc(nnIds(i),0);
  }
  mean /= (float)nnIds.rows();
  return mean;
}

tdp::Matrix3fda getCovariance(const tdp::Image<tdp::Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  // get covariance of the point cloud assuming no nan and pc of (nrows,1) size.
  assert (pc.h_ == 1);
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
    tdp::ManagedHostImage<tdp::Vector3fda> pc(7,1);
    for (size_t i=0; i<pc.Area(); ++i){
        tdp::Vector3fda pt;
        pt << i,0,0;
        pc(i,0) = pt;
    }
    return pc;
}


void GetSphericalPc(tdp::ManagedHostImage<tdp::Vector3fda>& pc){
    pc.Reinitialise(pc.w_, pc.h_);
    for (size_t i=0; i<pc.w_; ++i) {
       pc[i] = tdp::S3f::Random().vector();
    }
}

void GetCylindricalPc(tdp::ManagedHostImage<tdp::Vector3fda>& pc){
    //todo: use [s1;R]
    pc.Reinitialise(pc.w_, pc.h_);
    for (size_t i=0; i<pc.Area(); ++i){
        tdp::S2f pt_2d = tdp::S2f::Random().vector();
        float z  = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
        pc[i] = tdp::Vector3fda(pt_2d.vector()(0), pt_2d.vector()(1), z);
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

void getAllLocalBasis(const tdp::Image<tdp::Vector3fda>& pc, tdp::Image<tdp::SE3f>& T_wl,
                      tdp::ANN& ann, int knn, float eps){

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

inline tdp::Vector6fda poly2Basis(const tdp::Vector3fda& p){
    tdp::Vector6fda newP;
    newP << 1, p[0], p[1], p[0]*p[0], p[0]*p[1], p[1]*p[1];
    return newP;
}

inline Eigen::Vector4f homogeneous(const tdp::Vector3fda& p){
    return tdp::Vector4fda(p(0),p(1),p(2),1);
}

void getThetas(const tdp::Image<tdp::Vector3fda>& pc_w,
               const tdp::Image<tdp::SE3f>& T_wls, tdp::Image<tdp::Vector6fda>& thetas,
               tdp::ANN& ann, int knn, float eps){
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
            tdp::Vector3fda npt_l = T_wl.Inverse()*pc_w[nnIds[k]];
            //target is the third dim coordinate
            float npt_z = npt_l(2);
            //project to higher dimension using poly2 basis
            tdp::Vector6fda phi_npt = poly2Basis(npt_l);
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
    tdp::Vector3fda pt_l;
    tdp::Vector6fda phi_pt, theta;
    float z_estimated;
    for (size_t i=0; i<pc_w.Area(); ++i){
        pt_l = T_wl[i].Inverse()*pc_w[i];
        theta = thetas[i];
        //Estimate normals
        phi_pt = poly2Basis(pt_l);
        z_estimated = theta.transpose()*phi_pt;\
        estimates_w[i] = T_wl[i]*(tdp::Vector3fda(pt_l(0),pt_l(1),z_estimated));
   }
}

void getSamples(const tdp::Image<tdp::SE3f>& T_wl,
                const tdp::Image<tdp::Vector6fda>& thetas,
                tdp::Image<tdp::Vector3fda>& estimates_w, size_t upsample){
    tdp::Vector3fda pt_l;
    tdp::Vector6fda phi_pt;
    float z_estimated;
    for (size_t i=0; i<T_wl.Area(); ++i){
        for (size_t j=0; j<upsample; ++j) {
            pt_l = 0.1*tdp::Vector3fda::Random();
            //Estimate normals
            phi_pt = poly2Basis(pt_l);
            z_estimated = thetas[i].transpose()*phi_pt;\
            estimates_w[i*upsample+j] = T_wl[i]*(tdp::Vector3fda(pt_l(0),pt_l(1),z_estimated));
        }
   }
}

void getThetas_F(const tdp::Image<tdp::Vector3fda>& pc_w,
               const tdp::Image<tdp::SE3f>& T_wls, const auto& f,
               tdp::Image<tdp::Vector6fda>& thetas, tdp::ANN& ann, int knn, float eps){
    assert(pc_w.w_ == T_wls.w_&&pc_w.w_==thetas.w_);
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    for (size_t i=0; i<pc_w.Area(); ++i){
        tdp::Vector3fda pt = pc_w[i];
        const tdp::SE3f& T_wl = T_wls[i];

        // Get the neighbor ids and dists for this point
        ann.Search(pt, knn, eps, nnIds, dists);
        //std::cout << nnIds.transpose() << std::endl;

        tdp::MatrixXfda X(knn,6), W(knn,knn);//todo clean this up
        tdp::Vector3fda Y(knn);
        tdp::Vector6fda theta;
        for (size_t k=0; k<knn; ++k){
            //std::cout << "iter: " << k << std::endl;
            //std::cout << "kth neighbor pt in wc: \n" << pc(nnIds[k],0) <<std::endl;
//            if(knn!=10){
//                std::cout << "lenght of nnids: " << nnIds.rows() << std::endl;
//            }
//            std::cout <<"k: " << k << " " << knn << " " << nnIds.rows() << std::endl;
//            std::cout <<"nnids: " << nnIds(k) << std::endl;

            tdp::Vector3fda npt_l = T_wl.Inverse()*pc_w[nnIds(k)];
            //target of the weighted least square
            float y = f(npt_l);
            //std::cout << "z: " << npt_l(2) << std::endl;
            //std::cout << "using f: " << y << std::endl;
            //construct data matrix X
            X.row(k) = poly2Basis(npt_l);
            //construct target vector Y
            Y(k) = y;
            // weight matrix W
            W(k,k) = (dists(k)<1e-6? 1: 1.0f/knn);

//            //Take the first two dimensions
//            tdp::Vector2fda npt_2d(npt_l_(0), npt_l_(1));
//            //target is the third dim coordinate
//            float npt_z = f_z(npt_l);
//            //project to higher dimension using poly2 basis
//            tdp::Vector6fda phi_npt = poly2Basis(npt_2d);
//            //Construct data matrix X
//            X.row(k) = phi_npt;
//            //Construct target vector Y
//            Y(k) = npt_z;
//            //Get weight matrix W
//            W(k,k) = dists(k); //check if I need to scale this when in local coordinate system
        }

        //Solve weighted least square
        Eigen::FullPivLU<tdp::Matrix6fda> X_lu;
        X_lu.compute(X.transpose()*W*X);
        theta = X_lu.solve(X.transpose()*W*Y);
        thetas[i] = theta;
    }
}


void getFEstimates(const tdp::Image<tdp::Vector3fda>& pc_w,
                   const tdp::Image<tdp::SE3f>& T_wls,
                   const tdp::Image<tdp::Vector6fda>& thetas,
                   tdp::Image<tdp::Vector3fda>& estimates_w){
    tdp::Vector3fda pt_l;
    tdp::Vector6fda phi_pt, theta;
    float estimate_l;
    for (size_t i=0; i<pc_w.Area(); ++i){
        pt_l = T_wls[i].Inverse()*pc_w[i];
        theta = thetas[i];
        //Estimate normals
        phi_pt = poly2Basis(pt_l);
        estimate_l = theta.transpose()*phi_pt;
        estimates_w[i] = T_wls[i]*(tdp::Vector3fda(pt_l(0),pt_l(1),estimate_l));
   }
}

Eigen::VectorXf real(Eigen::VectorXcf vec_c){
    Eigen::VectorXf vec_r(vec_c.size());
    for (int i=0; i<vec_c.size(); i++){
        vec_r(i)= vec_c(i).real();
    }
    return vec_r;
}

Eigen::SparseMatrix<float> getLaplacian(tdp::Image<tdp::Vector3fda>& pc,
                                        tdp::ANN& ann,
                                        const int knn,
                                        const float eps,
                                        float alpha){

    Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());
    Eigen::VectorXi nnIds(knn,1);
    Eigen::VectorXf dists(knn,1);
    float row_sum;
    L.reserve(Eigen::VectorXi::Constant(pc.Area(),2*knn)); //todo: better memory init
    for (int i=0; i<pc.Area(); ++i){
        ann.Search(pc[i], knn, eps, nnIds, dists);
        alpha = dists.maxCoeff();
        row_sum = (-dists.array()/alpha).exp().sum();
        for (int k=0; k<knn; ++k){
            //todo: changes coeffRef to insert (seg fault)
            if (i==nnIds(k)) {
                L.insert(i,nnIds(k)) = 1;//(-dists.array()/alpha).exp().sum();
            } else {
                L.insert(i,nnIds(k)) = -(exp(-dists(k)/alpha)/row_sum);
            }
        }
        // show the current row
        // http://eigen.tuxfamily.org/dox/group__TutorialSparse.html
//        for (Eigen::SparseMatrix<float>::InnerIterator it(L,i); it; ++it){
//            std::cout << "\n\nrow index: " << it.row() << std::endl;
//            std::cout << "col index should be i: " <<it.col() <<std::endl;
//            std::cout << "val:" << it.value() <<std::endl;
//        }
    }
    return L;
}

Eigen::VectorXf getLaplacianEvector(const tdp::Image<tdp::Vector3fda>& pc,
                                    const Eigen::SparseMatrix<float>& L,
                                    int idEv){
    // Construct matrix operation object using the wrapper class SparseGenMatProd
    Spectra::SparseGenMatProd<float> op(L);

    // Retrieve results
    Eigen::VectorXcf evalues;
    Eigen::VectorXcf evector_complex;
    // Construct eigen solver object, requesting the largest idEv number of eigenvalues
    Spectra::GenEigsSolver<float, Spectra::SMALLEST_REAL,
            Spectra::SparseGenMatProd<float> > eigs(&op, idEv+1, 2*(idEv+1)+1);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(1000,1e-10, Spectra::SMALLEST_REAL);

    if(eigs.info() == Spectra::SUCCESSFUL) {
        evalues = eigs.eigenvalues();
        evector_complex = eigs.eigenvectors().col(idEv);
        std::cout << "Eigenvalues found:\n" << evalues.real().transpose() << std::endl; //check first should be zero
    } else{
        std::cout << "failed" << std::endl;
    }
    return evector_complex.real();
}


Eigen::MatrixXf getMeanCurvature(const tdp::Image<tdp::Vector3fda>& pc,
                                 const Eigen::SparseMatrix<float>& L){
    Eigen::MatrixXf pc_vec(pc.Area(),3);
    for (int i=0; i<pc.Area(); ++i){
        pc_vec(i,0) = pc[i](0); //x coordinate
        pc_vec(i,1) = pc[i](1); //y coordinate
        pc_vec(i,2) = pc[i](2);
    }
    return (Eigen::MatrixXf)L*pc_vec;//dense matrix

}

std::vector<tdp::Vector3fda> getLevelSetMeans(const tdp::Image<tdp::Vector3fda>& pc,
                                             const Eigen::VectorXf& evector,
                                             int nBins){

    float minV, maxV, step;
    minV = evector.minCoeff();
    maxV = evector.maxCoeff();
    step = (maxV - minV)/nBins;
    //std::cout << "minV, maxV, step: " << minV << ", " << maxV << ", " << step << std::endl;
    std::vector<tdp::Vector3fda> bins;
    bins.reserve(nBins);
    //Initialize bins with zero vectors
    for (int i=0; i<nBins; ++i){
        bins.push_back(tdp::Vector3fda(0,0,0));
    }

    std::vector<int> counts(nBins, 0);

    for (int i=0; i<evector.rows(); ++i){
        int bId = std::floor((evector(i)-minV)/step);
        if (evector(i)==maxV){
            bId -= 1;
        }
        bins[bId] += pc[i];
        counts[bId] += 1;
    }

    for (int i=0; i<bins.size(); ++i){
        if(counts[i]!=0){
            bins[i] /= (float)counts[i];
        }
    }

    return bins;
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
    tdp::Vector3fda vec1(10.,10.,10.);
    tdp::Vector3fda vec2(0,0,0);
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

void test_getThetas_F(){

}

void test_real(){
    std::complex<float> one_c(1.0f,1.0f);
    std::complex<float> two_c(2.0f,2.0f);
    std::complex<float> three_c(3.0f,3.0f);
    Eigen::Vector3cf vec_c;
    vec_c(0) = one_c;
    vec_c(1) = two_c;
    vec_c(2) = three_c;
    std::cout << real(vec_c) << std::endl;
}

void test_Laplacian(){
    tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();

    tdp::ANN ann;
    ann.ComputeKDtree(pc);

    Eigen::VectorXf evector(pc.Area(),1);
    Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());
    float alpha = 1;
    int idEv = 0; int knn=5; float eps=1e-6;
    L = getLaplacian(pc,ann,knn,eps,alpha);
    std::cout << "L: \n" << L << std::endl;
    evector = getLaplacianEvector(pc, L, idEv);
    std::cout << "evector: \n" << evector.transpose() << std::endl;

    //Test meancurvature
    Eigen::MatrixXf curvature(pc.Area(),3);
    curvature = getMeanCurvature(pc, L);
    std::cout << "mean curvature: \n" << curvature <<std::endl;
    //should be all zeros for a linear pc

    //Test getLevelSets
    int nBins = 2;
    std::vector<tdp::Vector3fda> means;
    means = getLevelSetMeans(pc, evector, nBins);
    std::cout << "means----" << std::endl;
    for (int i = 0; i< means.size(); ++i){
        std::cout << means[i] << std::endl;
    }
}

void test_getCylinder(){
    tdp::ManagedHostImage<tdp::Vector3fda> pc(10,1);
    GetCylindricalPc(pc);
}

//end of test

int main( int argc, char* argv[] ){
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
  pangolin::Var<int> knn("ui.knn",50,1,100);
  pangolin::Var<float> eps("ui.eps", 1e-6 ,1e-7, 1e-5);

  pangolin::Var<int> idEv("ui.id EV", 1, 0, 10);
  pangolin::Var<float> alpha("ui. alpha", 0.01, 0.005, 0.3); //variance of rbf kernel
  // viz color coding
  pangolin::Var<float>minVal("ui. min Val",-0.71,-1,0);
  pangolin::Var<float>maxVal("ui. max Val",0.01,1,0);
  pangolin::Var<int>nBins("ui. nBins", 50, 10,1000);
  // sampling
  pangolin::Var<int> upsample("ui.upsample", 10,1,100);

  tdp::ManagedHostImage<tdp::SE3f> T_wls(pc.w_,1);
  //tdp::ManagedHostImage<tdp::Vector6fda> thetas(pc.w_,1);
  //tdp::ManagedHostImage<tdp::Vector3fda> zEstimates(pc.w_,1),fEstimates(pc.w_,1);
  //tdp::ManagedHostImage<tdp::Vector3fda> zSamples(pc.w_*upsample,1);
  Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());
  Eigen::VectorXf evector(pc.Area(),1);
  Eigen::MatrixXf curvature(pc.Area(),3);

  std::vector<tdp::Vector3fda> means;
  means.reserve(nBins);
  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (pangolin::Pushed(runSkinning) || knn.GuiChanged() || upsample.GuiChanged()
            || idEv.GuiChanged() || alpha.GuiChanged() || nBins.GuiChanged()) {
        //  processing of PC for skinning
      std::cout << "Running skinning..." << std::endl;

      if( pcOption.GuiChanged()){
        std::cout << "Roading new pc..." << pcOption << std::endl;
        if (pcOption == 0){
            GetSphericalPc(pc);
        } else if (pcOption ==1 ){
            GetCylindricalPc(pc);
        }
      }

     // getAllLocalBasis(pc, T_wls, ann, knn, eps);
//      getThetas(pc,T_wls,thetas,ann,knn,eps);
//      getZEstimates(pc,T_wls,thetas,zEstimates);
//      vboZ.Reinitialise(pangolin::GlArrayBuffer, zEstimates.Area() , GL_FLOAT, 3, GL_DYNAMIC_DRAW ); //will later be knn*pc.Area()
//      vboZ.Upload(zEstimates.ptr_, sizeof(tdp::Vector3fda) * zEstimates.Area(), 0);

      //getThetas_F(pc, T_wls, f_z, thetas, ann, knn, eps);
      //getFEstimates(pc, T_wls, thetas, fEstimates);
      //vboF.Reinitialise(pangolin::GlArrayBuffer, fEstimates.Area() , GL_FLOAT, 3, GL_DYNAMIC_DRAW ); //will later be knn*pc.Area()
      //vboF.Upload(fEstimates.ptr_, sizeof(tdp::Vector3fda) * fEstimates.Area(), 0);

      // get Laplacian operator and its eigenvectors
      L = getLaplacian(pc, ann, knn, eps, alpha);
      evector = getLaplacianEvector(pc, L, idEv);
      //curvature = getMeanCurvature(pc, L);
//      // debug getMeanCurvature
//      for (int i=0; i<(int)curvature.rows(); ++i){
//          if(curvature.row(i)(0)>1e-7 && curvature.row(i)(1)>1e-7 && curvature.row(i)(2)>1e-7){
//            std::cout << "i: " << i << ", " <<curvature.row(i) << std::endl;
//          }
//      }
      //Decompose curvature to get normals and mean curvature value
      //Eigen::VectorXf meanCurvature = curvature.rowwise().norm();
      //std::cout << "meanCurvature of a row: " << meanCurvature.transpose() << std::endl;

      // Get the means of the level sets
      means = getLevelSetMeans(pc, evector, nBins);
      for (int i=0; i<means.size(); ++i){
          if(means[i](0)>1 || means[i](1) >1 || means[i](2)>2){
              std::cout << "OH NO! this should never be printed!: \n" << means[i](0) << std::endl;
          }
      }
      std::cout << std::endl;


      valuebo.Reinitialise(pangolin::GlArrayBuffer, evector.rows() ,
          GL_FLOAT,1, GL_DYNAMIC_DRAW);
      // TODO: upload values to visualize here
      valuebo.Upload(&evector(0), sizeof(float)*evector.rows(), 0);
      std::cout << evector.minCoeff() << " " << evector.maxCoeff() << std::endl;

      minVal = evector.minCoeff()-1e-3;
      maxVal = evector.maxCoeff();

//      zSamples.Reinitialise(pc.w_*upsample,1);
//      getSamples(T_wls,thetas,zSamples,upsample);

      // put the estimated points to GLBuffer vboM
      //vboM.Reinitialise(pangolin::GlArrayBuffer, means.size() , GL_FLOAT, 1, GL_DYNAMIC_DRAW );
      //vboM.Upload(&means[0], sizeof(float)*means.size(), 0);
      // Draw lines connecting the centers

      std::cout << "<--DONE skinning-->" << std::endl;
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
          laplace::glDrawLine(m_prev, m);
      }

      glPointSize(1.);
      // draw the first arm pc
      glColor3f(1.0f, 0.0f, 0.0f);
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
  }

  std::cout << "good morning!" << std::endl;
  return 0;
}