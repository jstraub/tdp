#include <tdp/testing/testing.h>
#include <tdp/laplace_beltrami/laplace_beltrami.h>

using namespace tdp;

TEST(laplace_beltrami, meanAndCov) {
    //TEST OF getMean and getCovariance
    tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();
    Eigen::VectorXi nnIds(10);
    nnIds<< 0,1,2,3,4,5,6,7,8,9;
    tdp::Vector3fda mean = getMean(pc, nnIds);
    tdp::Matrix3fda cov = getCovariance(pc,nnIds);
    std::cout << "mean: \n" << mean << std::endl << std::endl;
    std::cout << "cov: \n" << cov << std::endl << std::endl;
}

TEST(laplace_beltrami, getAllLocalBasis) {

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

//TEST(laplace_beltrami, testGetAxesIds) {

//    std::vector<int> v = {1,5,3};
//    std::vector<int> ids;
//    getAxesIds(v,ids);
//    for (int i =0; i<ids.size(); ++i){
//        std::cout << ids[i] << ": "<< v[ids[i]] << std::endl;
//    }
//}

TEST(laplace_beltrami, poly2Basis) {


    tdp::Vector3fda vec1(10.,10.,10.);
    tdp::Vector3fda vec2(0,0,0);
    std::cout << poly2Basis(vec1) << std::endl;
    std::cout << poly2Basis(vec2) << std::endl;

}

TEST(laplace_beltrami, getLocalRot) {

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


TEST(laplace_beltrami, Laplacian) {

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
    tdp::eigen_vector<tdp::Vector3fda> means;
    means = getLevelSetMeans(pc, evector, nBins);
    std::cout << "means----" << std::endl;
    for (int i = 0; i< means.size(); ++i){
        std::cout << means[i] << std::endl;
    }
}


TEST(laplace_beltrami, LaplacianEvectors) {
    float alpha = 1;
    int idEv = 0; int knn=5; float eps=1e-6;
    int numEv = 3;

    tdp::ManagedHostImage<tdp::Vector3fda> pc = GetSimplePc();

    tdp::ANN ann;
    ann.ComputeKDtree(pc);

    tdp::eigen_vector<Eigen::VectorXf> evectors(numEv, Eigen::VectorXf(pc.Area(),1));
    Eigen::VectorXf evector(pc.Area(),1);
    Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());

    L = getLaplacian(pc,ann,knn,eps,alpha);
    std::cout << "L: \n" << L << std::endl;

    getLaplacianEvectors(L, numEv, evectors);
    std::cout << "test evector: \n" << evectors[idEv] << std::endl;

    evector = getLaplacianEvector(pc, L, idEv);
    std::cout << "evector: \n" << evector.transpose() << std::endl;

}

TEST(laplace_beltrami, getCylinder) {

    tdp::ManagedHostImage<tdp::Vector3fda> pc(10,1);
    GetCylindricalPc(pc);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

