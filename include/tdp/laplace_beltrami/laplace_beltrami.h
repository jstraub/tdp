#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>

#include <Eigen/Eigenvalues>

#include <pangolin/gl/gldraw.h>


#include <tdp/eigen/dense.h>
#include <tdp/eigen/std_vector.h>
#include <tdp/data/managed_image.h>
#include <tdp/nn/ann.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

Vector3fda getMean(const Image<Vector3fda> &pc, const Eigen::VectorXi& nnIds);

Matrix3fda getCovariance(const Image<Vector3fda>& pc,
                         const Eigen::VectorXi& nnIds);

/*************Get Point Clouds****************************
**********************************************************/
/*
 * Get point clouds
 * shapeOpt: int, string pair for shapeOpt index and path to mesh file
         : (0,""), (1,""), (2,fpathToBunny), (3, fpathToManekine)
 */
void GetPc(const ManagedHostImage<Vector3fda>& pc,
           const std::pair<int, std::string>& shapeOpt,
           int nSamples);

void GetSimplePc(ManagedHostImage<Vector3fda>& pc, int n=10);

//Get points on sphere in spherical coordinates
void GetPointsOnSphere(ManagedHostImage<Vector3fda>& pc,
                      int nPoints=10,
                      float r=1.0f);

void GetSphericalPc(ManagedHostImage<Vector3fda>& pc,
                    int nSamples=10);

void GetCylindricalPc(ManagedHostImage<Vector3fda>& pc);

void GetCylindricalPc(ManagedHostImage<Vector3fda>& pc,
                      int nSamples);

void GetGrid(ManagedHostImage<Vector3fda>& pc, int w, int h, float step=0.1);


void GetSamples(const Image<tdp::Vector3fda>& pc,
                ManagedHostImage<Vector3fda>& samples,
                int nSamples);

void GetSamples_seed(const Image<Vector3fda>& pc,
                    ManagedHostImage<Vector3fda>& samples,
                    int nSamples,
                    std::random_device rd/*seed*/);

void addGaussianNoise(const ManagedHostImage<Vector3fda>& src,
                      float std,
                      ManagedHostImage<Vector3fda>& dst);
/*********Coordinate System****************************************************/
void toCartisean(const ManagedHostImage<Vector3fda>& pc_spherical,
                 ManagedHostImage<Vector3fda>& pc_cartisean);

/*********DEFORMATION*********************************************************/
void scale(const ManagedHostImage<Vector3fda>& src,
           const float factor,
           ManagedHostImage<Vector3fda>& dst);

std::vector<int> Deform(const ManagedHostImage<Vector3fda>& src,
                                ManagedHostImage<Vector3fda>& dst,
                                float max_phi);

/*****************************************************************************/
template<typename T>
inline void getAxesIds(const std::vector<T>& vec, std::vector<int>& sortIds){

    int hi(0), lo(0), mid;
    for (size_t i=0; i<vec.size(); ++i){
        if (vec[i] < vec[lo]){
            lo = i;
        } else if (vec[i] > vec[hi]){
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
};

Eigen::Matrix3f getLocalRot(const Matrix3fda& cov,
                            const Eigen::SelfAdjointEigenSolver<Matrix3fda>& es);

void getAllLocalBasis(const Image<Vector3fda>& pc, Image<SE3f>& locals,
                      ANN& ann, int knn, float eps);

inline float w(float d, int knn){
    return d==0? 1: 1/(float)knn;
}

inline Vector6fda poly2Basis(const Vector3fda& p){
    Vector6fda newP;
    newP << 1, p[0], p[1], p[0]*p[0], p[0]*p[1], p[1]*p[1];
    return newP;
}
void getThetas(const Image<Vector3fda>& pc, const Image<SE3f>& T_wls,
               Image<Vector6fda>& thetas, ANN& ann, int knn, float eps);

void getZEstimates(const Image<Vector3fda>& pc, const Image<SE3f>& locals,
                   const Image<Vector6fda>& thetas, Image<Vector3fda>& estimates);

void getThetas_F(const Image<Vector3fda>& pc_w,const Image<SE3f>& T_wls,
                 float (&f)(const Vector3fda&), Image<Vector6fda>& thetas, ANN& ann,
                 int knn, float eps);

void getFEstimates(const Image<Vector3fda>& pc_w,
                   const Image<SE3f>& T_wls,
                   const Image<Vector6fda>& thetas,
                   Image<Vector3fda>& estimates_w);

void  getSimpleLBEvector(Image<Vector3fda>& pc,
                         Eigen::VectorXf& evector_real,
                         ANN& ann,
                         const int knn,
                         const float eps,
                         float alpha,
                         int idEv);

Eigen::SparseMatrix<float> getLaplacian(Image<Vector3fda>& pc,
                                         ANN& ann,
                                         const int knn,
                                         const float eps,
                                         float alpha);

//template<typename T>
void showSparseMatrix(const Eigen::SparseMatrix<float>& S,
                      const int nCols=10,
                      const int nRows=10);


Eigen::VectorXf getLaplacianEvector(const Image<Vector3fda>& pc,
                                    const Eigen::SparseMatrix<float>& L,
                                    int idEv);

void getLaplacianEvectors(const Eigen::SparseMatrix<float>& L,
                          int numEv,
                          eigen_vector<Eigen::VectorXf>& evectors);

void getLaplacianBasis(const Eigen::SparseMatrix<float>& L,
                       int numEv,
                       Eigen::MatrixXf& basis);

void decomposeLaplacian(const Eigen::SparseMatrix<float>& L,
                        int numEv,
                        Eigen::VectorXf& evalues,
                        Eigen::MatrixXf& evectors
                        );

void gramSchmidt(Eigen::MatrixXf& basis);


Eigen::MatrixXf getMeanCurvature(const Image<Vector3fda>& pc,
                                 const Eigen::SparseMatrix<float>& L);


eigen_vector<Vector3fda> getLevelSetMeans(const Image<Vector3fda>& pc,
                                          const Eigen::VectorXf& evector,
                                          int nBins);



/******************CORRESPONDENCES****************************************
 *************************************************************************/
inline float rbf(const Vector3fda& p1,
                 const Vector3fda& p2,
                 const float alpha);


void f_rbf(const Image<Vector3fda>& pc,
           const Vector3fda& p,
           const float alpha,
           Eigen::VectorXf& f );


void f_height(const Image<Vector3fda>& pc,
              Eigen::VectorXf& f_w);

void f_indicator(const Image<Vector3fda>& pc,
                 const int p_idx,
                 Eigen::VectorXf& f);

void f_indicator(const Image<Vector3fda> &pc,
                 const std::vector<int>& support,
                 Eigen::VectorXf &f);

void f_landmark(const Image<Vector3fda>& pc,
                const int p_idx,
                const float alpha,
                const std::string& option,
                Eigen::VectorXf& f_w);

Eigen::MatrixXf getHKS(const Eigen::MatrixXf& LB_evecs,
                       const Eigen::VectorXf& LB_evals,
                       const int nSteps);

/******************print**********************************
**********************************************************/
void printImage(const ManagedHostImage<Vector3fda>& pc,
                int start_idx,
                int length);

/*****************fmap matrix clean up********************
**********************************************************/
//Fmap clean up to one or zero if the entry is close enough
inline void clean_near_zero_one(Eigen::MatrixXf& M,
                                const float threshold){
  int nOnes(0), nZeros(0);
  for (int i=0; i<M.rows(); ++i){
    for (int j=0; j<M.cols(); ++j){
      if (1-threshold <M(i,j) && M(i,j) < 1+threshold){
        M(i,j) = 1.f;
        nOnes += 1;
      } else if (-threshold <M(i,j) && M(i,j) < threshold){
        M(i,j) = 0.0f;
        nZeros += 1;
      }
    }
  }
  std::cout << "updated to one, zero: " << nOnes << ", " << nZeros 
            << std::endl;
}

/******************cache**********************************
**********************************************************/
std::map<std::string, std::string> makeCacheNames(
        const int shapeOpt, 
        const int nSamples, 
        const int knn, 
        const float alpha,
        const float noiseStd,
        const int nEv, 
        const char* cacheDir="./cache/");


/**************************PROJECTIONS***************************************
 ****************************************************************************/
// template <typename Derived, typename OtherDerived>
// inline Eigen::MatrixBase<OtherDerived> projectToLocal(
//                                        Eigen::MatrixBase<Derived>* T_wl,
//                                        const Eigen::MatrixBase<OtherDerived>& F_w)
// {
//     /*
//      * T_wl: each column is an eigenvector. n by k 
//      * (each vector lives in n-dim space. There are k evectors.)
//      * F_wl: each column is a vector in n-dim. Say F_wl.cols() is t
//      * Returns t vectors in k-dim space. k by t
//     */
//     return (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*F_w);
// };

// template <typename Derived, typename OtherDerived>
// inline Eigen::MatrixBase<OtherDerived> projectToWorld(
//                                         Eigen::MatrixBase<Derived>* T_wl,
//                                         Eigen::Ref<const Eigen::MatrixBase<OtherDerived> >& F_l)
// {
//     /*
//      * T_wl: each column is an eigenvector. n by k 
//      * (each vector lives in n-dim space. There are k evectors.)
//      * F_wl: each column is a vector in n-dim. Say F_wl.cols() is t
//      * Returns t vectors in k-dim space. k by t
//     */
//     return T_wl*F_l;
// };

// template <typename Derived, typename OtherDerived>
// inline Eigen::MatrixBase<OtherDerived> projectToLocal(
//                                        Eigen::MatrixBase<Derived>& T_wl,
//                                        Eigen::MatrixBase<OtherDerived>& F_w)
// {
//     /*
//      * T_wl: each column is an eigenvector. n by k 
//      * (each vector lives in n-dim space. There are k evectors.)
//      * F_wl: each column is a vector in n-dim. Say F_wl.cols() is t
//      * Returns t vectors in k-dim space. k by t
//     */
//     Eigen::MatrixXf product = (Eigen::MatrixXf)T_wl.transpose()*T_wl;
//     return product.fullPivLu().solve(T_wl.transpose()*F_w);
// };

// template <typename Derived, typename OtherDerived>
// inline Eigen::MatrixBase<OtherDerived> projectToWorld(
//                                         Eigen::MatrixBase<Derived>& T_wl,
//                                         const Eigen::MatrixBase<OtherDerived>& F_l)
// {
//     /*
//      * T_wl: each column is an eigenvector. n by k 
//      * (each vector lives in n-dim space. There are k evectors.)
//      * F_wl: each column is a vector in n-dim. Say F_wl.cols() is t
//      * Returns t vectors in k-dim space. k by t
//     */
//     return T_wl*F_l;
// };

}
