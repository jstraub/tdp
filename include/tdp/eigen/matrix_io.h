#pragma once
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

namespace tdp  {

template <typename Matrix>
void write_binary(const char* filename, const Matrix& M){
	std::ofstream ofs(filename, std::ios::out | std::ios::binary | 
								std::ios::trunc);
	typename Eigen::Index  rows = M.rows(),
						   cols = M.cols();

	ofs.write((char*)(&rows), sizeof(typename Eigen::Index));
	ofs.write((char*)(&cols), sizeof(typename Eigen::Index));
	ofs.write((char*) M.data(), rows*cols*sizeof(typename Matrix::Scalar) );
	ofs.close();
}

template <typename Matrix>
void read_binary(const char* filename, Matrix& M){
	std::cout << "fname: " << filename << std::endl;
	std::ifstream ifs(filename, std::ios::in | std::ios::binary );
	if(ifs.is_open()){
		typename Eigen::Index rows = 0, cols = 0;
		ifs.read((char*) (&rows), sizeof(typename Eigen::Index));
		ifs.read((char*) (&cols), sizeof(typename Eigen::Index));
		std::cout << "rows, cols: " << rows << ", " << cols << std::endl; 
		M.resize(rows, cols);

		ifs.read((char*) M.data(), rows*cols*sizeof(typename Matrix::Scalar) );
		ifs.close();
		std::cout << "File read." << std::endl;
	} else{ 
		std::cout << "Unable to open file" << std::endl;
	}
}


} // write_binary, read_binary for Eigen::Matrix
