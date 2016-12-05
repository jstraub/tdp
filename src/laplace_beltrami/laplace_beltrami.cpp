#include <tdp/laplace_beltrami/laplace_beltrami.h>
#include <tdp/manifold/S.h>

namespace tdp {

float f_z(const Vector3fda& x) {
    return x(2);
}
\
float f_etoz(const Vector3fda& x){
//    return (float)exp(x(2));
    return x(2)*x(2);
}

Vector3fda getMean(const Image<Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  assert(pc.h_ == 1);
  Vector3fda mean(0,0,0);
  for (size_t i=0; i<nnIds.rows(); ++i){
      mean +=  pc(nnIds(i),0);
  }
  mean /= (float)nnIds.rows();
  return mean;
}

Matrix3fda getCovariance(const Image<Vector3fda>& pc, const Eigen::VectorXi& nnIds){
  // get covariance of the point cloud assuming no nan and pc of (nrows,1) size.
  assert (pc.h_ == 1);
  Matrix3fda cov;
  cov.setZero(3,3);

  Vector3fda mean = getMean(pc, nnIds);
  for(size_t i=0; i<nnIds.rows(); ++i){
    cov += (pc(nnIds(i),0)-mean)*(pc(nnIds(i),0)-mean).transpose();
  }
  cov /= (float)nnIds.rows();
  return cov;
}

ManagedHostImage<Vector3fda> GetSimplePc(){
    ManagedHostImage<Vector3fda> pc(10,1);
    for (size_t i=0; i<pc.Area(); ++i){
        Vector3fda pt;
        pt << i+1,0,0;
        pc[i] = pt;
    }
    return pc;
}


void GetSphericalPc(ManagedHostImage<Vector3fda>& pc){
    //pc.Reinitialise(pc.w_, pc.h_);
    for (size_t i=0; i<pc.w_; ++i) {
       pc[i] = S3f::Random().vector();
    }
}

void GetSphericalPc(ManagedHostImage<Vector3fda>& pc,
                    int nSamples){
    pc.Reinitialise(nSamples,1);
    for (size_t i=0; i<nSamples; ++i) {
       pc[i] = S3f::Random().vector();
    }
}

void GetCylindricalPc(ManagedHostImage<Vector3fda>& pc){
    //todo: use [s1;R]
    //pc.Reinitialise(nSamples,1);
    for (size_t i=0; i<pc.w_; ++i){
        S2f pt_2d = S2f::Random().vector();
        float z  = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
        pc[i] = Vector3fda(pt_2d.vector()(0), pt_2d.vector()(1), z);
    }
}


void GetCylindricalPc(ManagedHostImage<Vector3fda>& pc,
                      int nSamples){
    //todo: use [s1;R]
    pc.Reinitialise(nSamples,1);
    for (size_t i=0; i<nSamples; ++i){
        S2f pt_2d = S2f::Random().vector();
        float z  = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
        pc[i] = Vector3fda(pt_2d.vector()(0), pt_2d.vector()(1), z);
    }
}

void GetMtxPc(tdp::ManagedHostImage<Vector3fda>& pc, int w, int h, float step){
    for(int r=0; r<h; ++r){
        for (int c=0; c<w; ++c){
            pc[r*w+c] = tdp::Vector3fda(c*step,r*step,0);
        }
    }
}

Eigen::Matrix3f getLocalRot(const Matrix3fda& cov, const Eigen::SelfAdjointEigenSolver<Matrix3fda>& es){

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

void getAllLocalBasis(const Image<Vector3fda>& pc, Image<SE3f>& T_wl,
                      ANN& ann, int knn, float eps){

    //assumes ANN has complete computing kd tree
    //query `knn` number of neighbors
    assert( (pc.w_==T_wl.w_)&&(pc.h_ == T_wl.h_) );

    Vector3fda query, localMean;
    Matrix3fda cov, localRot;
    Eigen::SelfAdjointEigenSolver<Matrix3fda> es;
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    for (size_t i = 0; i<pc.Area(); ++i){
        query = pc(i,0);
        ann.Search(query, knn, eps, nnIds, dists);
        cov = getCovariance(pc,nnIds);
        es.compute(cov);
        localRot = getLocalRot(cov,es);
        localMean = getMean(pc, nnIds);
        T_wl[i] = SE3f(localRot, localMean);
        //Progress(i,pc.Area());
    }
}

inline float w(float d, int knn){
    return d==0? 1: 1/(float)knn;
}

inline Vector6fda poly2Basis(const Vector3fda& p){
    Vector6fda newP;
    newP << 1, p[0], p[1], p[0]*p[0], p[0]*p[1], p[1]*p[1];
    return newP;
}

inline Eigen::Vector4f homogeneous(const Vector3fda& p){
    return Vector4fda(p(0),p(1),p(2),1);
}

void getThetas(const Image<Vector3fda>& pc_w,
               const Image<SE3f>& T_wls, Image<Vector6fda>& thetas,
               ANN& ann, int knn, float eps){
    assert(pc_w.w_ == T_wls.w_&&pc_w.w_==thetas.w_);
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    for (size_t i=0; i<pc_w.Area(); ++i){
        Vector3fda pt = pc_w[i];
        const SE3f& T_wl = T_wls[i];

        // Get the neighbor ids and dists for this point
        ann.Search(pt, knn, eps, nnIds, dists);

        MatrixXfda X(knn,6), W(knn,knn);//todo clean this up
        VectorXfda Y(knn);
        Vector6fda theta;
        for (size_t k=0; k<knn; ++k){
            //std::cout << "iter: " << k << std::endl;
            //std::cout << "kth neighbor pt in wc: \n" << pc(nnIds[k],0) <<std::endl;
            Vector3fda npt_l = T_wl.Inverse()*pc_w[nnIds[k]];
            //target is the third dim coordinate
            float npt_z = npt_l(2);
            //project to higher dimension using poly2 basis
            Vector6fda phi_npt = poly2Basis(npt_l);
            //Construct data matrix X
            X.row(k) = phi_npt;
            //Construct target vector Y
            Y(k) = npt_z;
            //Get weight matrix W
            W(k,k) = dists(k); //check if I need to scale this when in local coordinate system
        }

        //Solve weighted least square
        Eigen::FullPivLU<Matrix6fda> X_lu;
        X_lu.compute(X.transpose()*W*X);
        theta = X_lu.solve(X.transpose()*W*Y);
        thetas[i] = theta;
    }
}

void getZEstimates(const Image<Vector3fda>& pc_w,
                   const Image<SE3f>& T_wl,
                   const Image<Vector6fda>& thetas,
                   Image<Vector3fda>& estimates_w){
    Vector3fda pt_l;
    Vector6fda phi_pt, theta;
    float z_estimated;
    for (size_t i=0; i<pc_w.Area(); ++i){
        pt_l = T_wl[i].Inverse()*pc_w[i];
        theta = thetas[i];
        //Estimate normals
        phi_pt = poly2Basis(pt_l);
        z_estimated = theta.transpose()*phi_pt;\
        estimates_w[i] = T_wl[i]*(Vector3fda(pt_l(0),pt_l(1),z_estimated));
   }
}

void getSamples(const Image<SE3f>& T_wl,
                const Image<Vector6fda>& thetas,
                Image<Vector3fda>& estimates_w, size_t upsample){
    Vector3fda pt_l;
    Vector6fda phi_pt;
    float z_estimated;
    for (size_t i=0; i<T_wl.Area(); ++i){
        for (size_t j=0; j<upsample; ++j) {
            pt_l = 0.1*Vector3fda::Random();
            //Estimate normals
            phi_pt = poly2Basis(pt_l);
            z_estimated = thetas[i].transpose()*phi_pt;\
            estimates_w[i*upsample+j] = T_wl[i]*(Vector3fda(pt_l(0),pt_l(1),z_estimated));
        }
   }
}

void getThetas_F(const Image<Vector3fda>& pc_w,
               const Image<SE3f>& T_wls, float (&f)(const Vector3fda&),
               Image<Vector6fda>& thetas, ANN& ann, int knn, float eps){
    assert(pc_w.w_ == T_wls.w_&&pc_w.w_==thetas.w_);
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    for (size_t i=0; i<pc_w.Area(); ++i){
        Vector3fda pt = pc_w[i];
        const SE3f& T_wl = T_wls[i];

        // Get the neighbor ids and dists for this point
        ann.Search(pt, knn, eps, nnIds, dists);
        //std::cout << nnIds.transpose() << std::endl;

        MatrixXfda X(knn,6), W(knn,knn);//todo clean this up
        Vector3fda Y(knn);
        Vector6fda theta;
        for (size_t k=0; k<knn; ++k){
            //std::cout << "iter: " << k << std::endl;
            //std::cout << "kth neighbor pt in wc: \n" << pc(nnIds[k],0) <<std::endl;
//            if(knn!=10){
//                std::cout << "lenght of nnids: " << nnIds.rows() << std::endl;
//            }
//            std::cout <<"k: " << k << " " << knn << " " << nnIds.rows() << std::endl;
//            std::cout <<"nnids: " << nnIds(k) << std::endl;

            Vector3fda npt_l = T_wl.Inverse()*pc_w[nnIds(k)];
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
//            Vector2fda npt_2d(npt_l_(0), npt_l_(1));
//            //target is the third dim coordinate
//            float npt_z = f_z(npt_l);
//            //project to higher dimension using poly2 basis
//            Vector6fda phi_npt = poly2Basis(npt_2d);
//            //Construct data matrix X
//            X.row(k) = phi_npt;
//            //Construct target vector Y
//            Y(k) = npt_z;
//            //Get weight matrix W
//            W(k,k) = dists(k); //check if I need to scale this when in local coordinate system
        }

        //Solve weighted least square
        Eigen::FullPivLU<Matrix6fda> X_lu;
        X_lu.compute(X.transpose()*W*X);
        theta = X_lu.solve(X.transpose()*W*Y);
        thetas[i] = theta;
    }
}


void getFEstimates(const Image<Vector3fda>& pc_w,
                   const Image<SE3f>& T_wls,
                   const Image<Vector6fda>& thetas,
                   Image<Vector3fda>& estimates_w){
    Vector3fda pt_l;
    Vector6fda phi_pt, theta;
    float estimate_l;
    for (size_t i=0; i<pc_w.Area(); ++i){
        pt_l = T_wls[i].Inverse()*pc_w[i];
        theta = thetas[i];
        //Estimate normals
        phi_pt = poly2Basis(pt_l);
        estimate_l = theta.transpose()*phi_pt;
        estimates_w[i] = T_wls[i]*(Vector3fda(pt_l(0),pt_l(1),estimate_l));
   }
}

Eigen::VectorXf real(Eigen::VectorXcf vec_c){
    Eigen::VectorXf vec_r(vec_c.size());
    for (int i=0; i<vec_c.size(); i++){
        vec_r(i)= vec_c(i).real();
    }
    return vec_r;
}

Eigen::SparseMatrix<float> getLaplacian(Image<Vector3fda>& pc,
                                        ANN& ann,
                                        const int knn,
                                        const float eps,
                                        float alpha){

//    typedef Eigen::Triplet<double> T;
//    std::vector<T> tripletList;
//    tripletList.reserve(estimation_of_entries);

    Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());
    Eigen::VectorXi nnIds(knn);
    Eigen::VectorXf dists(knn);
    L.reserve(Eigen::VectorXi::Constant(pc.Area(),knn)); //todo: better memory init
    for (int i=0; i<pc.Area(); ++i){
        ann.Search(pc[i], knn, eps, nnIds, dists);
        alpha = dists.maxCoeff();
        float sum = (-dists.array()/alpha).exp().sum();
        for (int k=0; k<knn; ++k){
            //todo: changes coeffRef to insert (seg fault)
            if (i==nnIds(k)) {
                L.insert(i,nnIds(k)) = 1.;//(-dists.array()/alpha).exp().sum();
            } else {
                L.insert(i,nnIds(k)) = -exp(-dists(k)/alpha)/sum;
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

void getLaplacianEvectors(const Eigen::SparseMatrix<float>& L,
                          int numEv,
                          eigen_vector<Eigen::VectorXf>& evectors){

    assert(0<numEv && numEv<=L.rows());
    // Construct matrix operation object using the wrapper class SparseGenMatProd
    Spectra::SparseGenMatProd<float> op(L);

    // Retrieve results
    // Construct eigen solver object, requesting the largest idEv number of eigenvalues
    Spectra::GenEigsSolver<float, Spectra::SMALLEST_REAL,
            Spectra::SparseGenMatProd<float> > eigs(&op, numEv, 2*(numEv)+1);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(1000,1e-10, Spectra::SMALLEST_REAL);

    if(eigs.info() == Spectra::SUCCESSFUL) {
        for (int i=0; i<numEv; ++i){
            evectors[i] = eigs.eigenvectors().col(i).real();
            //std::cout << "i, evec: " << i << ", " << evectors[i].transpose() << std::endl;
        }
    } else{
        std::cout << "<----failed to get laplacian evectors--->" << std::endl;
    }
}

//todo: clean these functions (getLaplacianEvectors and getLaplacianBasis can be one function)
void getLaplacianBasis(const Eigen::SparseMatrix<float>& L,
                       int numEv,
                       Eigen::MatrixXf& basis){
    // Each row contains an evector
    // returns numEv by L.rows() matrix

    assert(0<numEv && numEv<=L.rows());
    // Construct matrix operation object using the wrapper class SparseGenMatProd
    Spectra::SparseGenMatProd<float> op(L);

    // Retrieve results
    // Construct eigen solver object, requesting the largest idEv number of eigenvalues
    Spectra::GenEigsSolver<float, Spectra::SMALLEST_REAL,
            Spectra::SparseGenMatProd<float> > eigs(&op, numEv, std::min(2*(numEv)+1, (int)L.rows()));

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(1000,1e-10, Spectra::SMALLEST_REAL);

    if(eigs.info() == Spectra::SUCCESSFUL) {
        for (int i=0; i<numEv; ++i){
            basis.col(i) = eigs.eigenvectors().col(i).real();
            //std::cout << "i, evec: " << i << ", " << evectors[i].transpose() << std::endl;
        }
    } else{
        std::cout << "<----failed to get laplacian evectors--->" << std::endl;
    }
}

inline Eigen::VectorXf getLaplacianEvector(const eigen_vector<Eigen::VectorXf>& evectors,
                                           int idEv){
    return evectors[idEv];
}

Eigen::VectorXf getLaplacianEvector(const Image<Vector3fda>& pc,
                                    const Eigen::SparseMatrix<float>& L,
                                    int idEv){
    // Construct matrix operation object using the wrapper class SparseGenMatProd
    Spectra::SparseGenMatProd<float> op(L);

    // Retrieve results
    Eigen::VectorXcf evalues(pc.Area());
    Eigen::VectorXcf evector_complex(pc.Area());
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
        evector_complex =  Eigen::VectorXcf::Zero(pc.Area());
    }
    return evector_complex.real();
}


Eigen::MatrixXf getMeanCurvature(const Image<Vector3fda>& pc,
                                 const Eigen::SparseMatrix<float>& L){
    Eigen::MatrixXf pc_vec(pc.Area(),3);
    for (int i=0; i<pc.Area(); ++i){
        pc_vec(i,0) = pc[i](0); //x coordinate
        pc_vec(i,1) = pc[i](1); //y coordinate
        pc_vec(i,2) = pc[i](2);
    }
    return (Eigen::MatrixXf)L*pc_vec;//dense matrix

}

eigen_vector<Vector3fda> getLevelSetMeans(const Image<Vector3fda>& pc,
                                          const Eigen::VectorXf& evector,
                                          int nBins){

    float minV, maxV, step;
    minV = evector.minCoeff();
    maxV = evector.maxCoeff();
    step = (maxV - minV)/nBins;
    //std::cout << "minV, maxV, step: " << minV << ", " << maxV << ", " << step << std::endl;
    eigen_vector<Vector3fda> bins(nBins, Vector3fda::Zero());
//    bins.reserve(nBins);
//    //Initialize bins with zero vectors
//    for (int i=0; i<nBins; ++i){
//        bins.push_back(Vector3fda(0,0,0));
//    }
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



void f_rbf(const Image<Vector3fda>& pc,
           const Vector3fda& p,
           const float alpha,
           Eigen::VectorXf& f){

    for (int i=0; i<pc.Area(); ++i){
        f(i) = (exp(-(1/alpha)*(pc[i]-p).squaredNorm()));
        //std::cout << "f(i):  " << f(i) << std::endl;
    }
    //std::cout << "f_rbf done" << std::endl;
}


}
