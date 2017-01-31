#include <tdp/testing/testing.h>
#include <tdp/eigen/dense_io.h>
#include <tdp/eigen/sparse_io.h>

using namespace tdp;


TEST(eigen_io, dense){
    Eigen::MatrixXi M,N;
    const char* fname = std::string("./cache/testio.dat").c_str();
    M.resize(3,3);
    M << 1,2,3,
         4,5,6,
         7,8,9;
    tdp::write_binary(fname, M);
    std::cout << "binary written to " << fname << std::endl;

    tdp::read_binary(fname, N);
    std::cout << N << std::endl;
    //assert(M == N);
}

TEST(eigen_io, sparse){
    Eigen::SparseMatrix<float> S(10,10), S_COPY(10,10);
    //S(0,0) = 10; S(1,1) = 20; S(2,2) = 30;
    for (int i=0; i<10; ++i){
        S.insert(0,i) = 10*(i+1);
    }

    tdp::write_binary("./cache/sparse.dat", S);
    std::cout << "Sparse written as bin---" << std::endl;

    std::cout << "Reading sparse---" << std::endl;
    tdp::read_binary("./cache/sparse.dat", S_COPY);
    for (int i=0; i<S_COPY.cols(); ++i){
        std::cout << S_COPY.col(i) << std::endl;
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

