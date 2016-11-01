#include <tdp/testing/testing.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <SymEigsSolver.h>

#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>


TEST(Spectra, dense) {
  // We are going to calculate the eigenvalues of M
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
  Eigen::MatrixXd M = A + A.transpose();

  // Construct matrix operation object using the wrapper class DenseGenMatProd
  Spectra::DenseSymMatProd<double> op(M);

  // Construct eigen solver object, requesting the largest three eigenvalues
  Spectra::SymEigsSolver< double, 
    Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > eigs(&op, 3, 6);

  // Initialize and compute
  eigs.init();
  int nconv = eigs.compute();

  // Retrieve results
  Eigen::VectorXd evalues;
  if(eigs.info() == Spectra::SUCCESSFUL)
    evalues = eigs.eigenvalues();

  std::cout << "Eigenvalues found:\n" << evalues << std::endl;
}

TEST(Spectra, sparse) {
  // A band matrix with 1 on the main diagonal, 2 on the below-main subdiagonal,
  // and 3 on the above-main subdiagonal
  const int n = 1000;
  Eigen::SparseMatrix<double> M(n, n);
  M.reserve(Eigen::VectorXi::Constant(n, 3));
  for(int i = 0; i < n; i++)
  {
    M.insert(i, i) = 1.0;
    if(i > 0)
      M.insert(i - 1, i) = 3.0;
    if(i < n - 1)
      M.insert(i + 1, i) = 2.0;
  }

  // Construct matrix operation object using the wrapper class SparseGenMatProd
  Spectra::SparseGenMatProd<double> op(M);

  // Construct eigen solver object, requesting the largest three eigenvalues
  Spectra::GenEigsSolver< double, Spectra::SMALLEST_MAGN,
    Spectra::SparseGenMatProd<double> > eigs(&op, 3, 6);

  // Initialize and compute
  eigs.init();
  int nconv = eigs.compute();

  // Retrieve results
  Eigen::VectorXcd evalues;
  if(eigs.info() == Spectra::SUCCESSFUL) {
    evalues = eigs.eigenvalues();
    std::cout << "Eigenvalues found:\n" << evalues << std::endl;
//    std::cout << eigs.eigenvectors() << std::endl;
  }

}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

