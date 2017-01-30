#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Sparse>
#include <typeinfo> 

namespace tdp  {

// typedef Eigen::Triplet<float> Triplet;
// typedef Eigen::SparseMatrix<float> SpMat;

// std::vector< Eigen::Triplet<float> > getTriplets(const Eigen::SparseMatrix<float>& S){
 void getTriplets(const Eigen::SparseMatrix<float>& S,
 				  std::vector<Eigen::Triplet<float>>& trips){
	int nCoeffs = S.nonZeros();
	trips.resize(nCoeffs);

	int tidx = 0;
	for (int k=0; k<S.outerSize(); ++k){
		for (Eigen::SparseMatrix<float>::InnerIterator it(S,k); it; ++it){
			trips[tidx++] = Eigen::Triplet<float>(it.row(),it.col(), it.value());
		}
	}
}

void write_binary(const char* filename, const Eigen::SparseMatrix<float>& S){
	std::ofstream ofs(filename, std::ios::out | std::ios::binary | 
								std::ios::trunc);

	for (int k=0; k<S.outerSize(); ++k){
		for(Eigen::SparseMatrix<float>::InnerIterator it(S,k); it; ++it){
			Eigen::Triplet<float> trip(it.row(), it.col(), it.value());
			ofs.write((char*) &trip, sizeof(trip));		
			std::cout << "triplet written" << std::endl;
		}
	}
	ofs.close();
}

void read_binary(const char*filename, Eigen::SparseMatrix<float>& S){
	std::ifstream ifs(filename, std::ios::in | std::ios::binary | std::ios::ate);

	if (ifs.is_open()){
		std::cout <<"\nfilename: " << filename << std::endl;

		const std::streampos FSIZE = ifs.tellg(); //Note we open from the end
		const int TSIZE = sizeof(Eigen::Triplet<float>);
		const int N = FSIZE/TSIZE; //num triplets
		std::cout << "nonzeros: " << N << std::endl;

		std::vector<Eigen::Triplet<float>> trips(N);
		ifs.seekg(0, std::ios::beg); //Move readptr to the start
		for (int i=0; i<N; ++i){
			Eigen::Triplet<float> trip;
			ifs.read((char*) &trip, sizeof(trip));
			trips[i] = trip;
		}
		ifs.close();
		std::cout << "Sparse Matrix is read." << std::endl;

		S.setFromTriplets(trips.begin(), trips.end());

	} else{
		std::cout << "Unable to open file." << std::endl;
	}

}
						


// template <typename SMat>
// void write_binary(const char* filename, const SMat& M){
// 	std::ofstream ofs(filename, std::ios::out | std::ios::binary | 
// 								std::ios::trunc);
// 	typename Eigen::Index  rows = M.rows(),
// 						   cols = M.cols();

// 	ofs.write((char*)(&rows), sizeof(typename Eigen::Index));
// 	ofs.write((char*)(&cols), sizeof(typename Eigen::Index));
// 	ofs.write((char*) M.data(), rows*cols*sizeof(typename Matrix::Scalar) );
// 	ofs.close();
// }

// template <typename Matrix>
// void read_binary(const char* filename, Matrix& M){
// 	std::cout << "fname: " << filename << std::endl;
// 	std::ifstream ifs(filename, std::ios::in | std::ios::binary );
// 	if(ifs.is_open()){
// 		typename Eigen::Index rows = 0, cols = 0;
// 		ifs.read((char*) (&rows), sizeof(typename Eigen::Index));
// 		ifs.read((char*) (&cols), sizeof(typename Eigen::Index));
// 		std::cout << "rows, cols: " << rows << ", " << cols << std::endl; 
// 		M.resize(rows, cols);

// 		ifs.read((char*) M.data(), rows*cols*sizeof(typename Matrix::Scalar) );
// 		ifs.close();
// 		std::cout << "File read." << std::endl;
// 	} else{ 
// 		std::cout << "Unable to open file" << std::endl;
// 	}
// }


} // write and read for Eigen::Sparse
