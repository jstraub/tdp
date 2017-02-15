#include <tdp/testing/testing.h>
#include <tdp/laplace_beltrami/laplace_beltrami.h>

using namespace tdp;

TEST(laplace_beltrami, meanAndCov) {
    //TEST OF getMean and getCovariance
    ManagedHostImage<Vector3fda> pc(10,1);
    GetSimplePc(pc);
    Eigen::VectorXi nnIds(10);
    nnIds<< 0,1,2,3,4,5,6,7,8,9;
    Vector3fda mean = getMean(pc, nnIds);
    Matrix3fda cov = getCovariance(pc,nnIds);
    std::cout << "mean: \n" << mean << std::endl << std::endl;
    std::cout << "cov: \n" << cov << std::endl << std::endl;
}

TEST(laplace_beltrami, getAllLocalBasis) {

    //test getAllLocalBasis
    ManagedHostImage<tdp::Vector3fda> pc(10,1);
    GetSimplePc(pc);
    ManagedHostImage<tdp::SE3f> locals(pc.w_,1);

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


    Vector3fda vec1(10.,10.,10.), vec2(0,0,0);
    std::cout << poly2Basis(vec1) << std::endl;
    std::cout << poly2Basis(vec2) << std::endl;

}

TEST(laplace_beltrami, getLocalRot) {

    ManagedHostImage<Vector3fda> pc(10,1);
    GetSimplePc(pc);
    Vector3fda query;
    Matrix3fda cov, localRot;
    Eigen::SelfAdjointEigenSolver<Matrix3fda> es;
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

    ManagedHostImage<Vector3fda> pc(10,1);
    GetSimplePc(pc);

    ANN ann;
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
    eigen_vector<Vector3fda> means;
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

    ManagedHostImage<tdp::Vector3fda> pc(10);
    GetSimplePc(pc);

    ANN ann;
    ann.ComputeKDtree(pc);

    eigen_vector<Eigen::VectorXf> evectors(numEv, Eigen::VectorXf(pc.Area(),1));
    Eigen::VectorXf evector(pc.Area(),1);
    Eigen::SparseMatrix<float> L(pc.Area(), pc.Area());

    L = getLaplacian(pc,ann,knn,eps,alpha);
    std::cout << "L: \n" << L << std::endl;

    getLaplacianEvectors(L, numEv, evectors);
    std::cout << "test evector: \n" << evectors[idEv] << std::endl;

    evector = getLaplacianEvector(pc, L, idEv);
    std::cout << "evector: \n" << evector.transpose() << std::endl;

    //--Test getLaplacianBasis
    std::cout << "numEv: " << numEv << std::endl;
    Eigen::MatrixXf basis(pc.Area(), numEv);
    getLaplacianBasis(L, numEv, basis);
    std::cout << "basis: \n" << basis << std::endl;

}

TEST(laplace_beltrami, getCylinder) {

    ManagedHostImage<tdp::Vector3fda> pc(10,1);
    GetCylindricalPc(pc);
}

TEST(laplace_beltrami, RbfKernels){
    ManagedHostImage<tdp::Vector3fda> pc(10,1);
    GetSimplePc(pc);
//    std::vector<float> f;
//    float alpha = 0.1;

//    std::cout << "RBF kernel..." << std::endl;
//    f = f_rbf(pc, pc[0], alpha);
//    for (int i=0; i < f.size(); ++i){
//        std::cout << f[i] << std::endl;
//    }

    float alpha = 0.1;
    //Eigen::VectorXf f(pc.Area()); //q: why can't i declare it here and assign two lines later??
    std::cout << "RBF kernel..." << std::endl;
    Eigen::VectorXf f(pc.Area());
    f_rbf(pc, pc[0], alpha,f);
    //std::cout << f.transpose() << std::endl; //q: why segfault???

}

TEST(laplace_beltrami, randomSeed){
  ManagedHostImage<tdp::Vector3fda> pc1, pc2, pc3, pc4;
  GetSphericalPc(pc1, 10);
  GetSphericalPc(pc2, 10);
  GetSphericalPc(pc3, 10);
  pc4.ResizeCopyFrom(pc3);

  std::cout << "PC1---" << std::endl;
  printImage(pc1, 0, pc1.Area());
  std::cout << "PC2---" << std::endl;
  printImage(pc2, 0, pc2.Area());
  std::cout << "PC3---" << std::endl;
  printImage(pc3, 0, pc3.Area());
  std::cout << "PC4---" << std::endl;
  printImage(pc4, 0, pc4.Area());
}

TEST(laplace_beltrami, f_landmark){
    std::string opt;
    float alpha = 0.1;
    ManagedHostImage<Vector3fda> pc(10,1);
    GetSimplePc(pc);
    Eigen::VectorXf f_w;

    for (int p_idx = 0; p_idx < pc.Area(); p_idx++){
        opt = "rbf";
        f_landmark(pc, p_idx, alpha, opt, f_w);
        std::cout << "opt: " << opt << std::endl;
        std::cout << "fw: " << f_w.transpose() << std::endl;

        opt = "ind";
        f_landmark(pc, p_idx, alpha, opt, f_w);
        std::cout << "opt: " << opt << std::endl;
        std::cout << "fw: " << f_w.transpose() << std::endl;
        std::cout << "\n\n" << std::endl;
    }
}


TEST(laplace_beltrami, printImage){
    ManagedHostImage<Vector3fda> pc(10,1);
    GetSimplePc(pc);
    std::cout << "---tdp::ing image---" << std::endl;
    printImage(pc,0,pc.Area());

    pc.Reinitialise(10);
    for (int i =0; i< 10; ++i){
        pc[i] = Vector3fda(i,i,i);
    }
    std::cout << "---Printing image---" << std::endl;
    printImage(pc,0,pc.Area());
}


TEST(laplace_beltrami, addGaussianNoise){
   ManagedHostImage<Vector3fda> pc_s, pc_t;
   GetSimplePc(pc_s);
   addGaussianNoise(pc_s, 0.1f, pc_t);

   std::cout << "PC_S ----" << std::endl;
   printImage(pc_s, 0, pc_s.Area());
   std::cout << "PC_T ---" << std::endl;
   printImage(pc_t, 0, pc_t.Area());
}

TEST(laplace_beltrami, clean_near_zero_one){
  Eigen::MatrixXf M(3,3);
  M << 1.0f, 1.001f, 0.99991,
       0.0f, 0.0001f, -0.0001f,
       0.0f, 1.0f, 1.0f;
  clean_near_zero_one(M, 0.002);
  std::cout << M << std::endl;
}

TEST(laplace_beltrami, makeCacheNames){
  int shapeOpt = 0;
  int nSamples = 10;
  int knn = 10;
  float alpha = 0.01;
  float noiseStd = 0.001;
  int nEv = 5;

  std::map<std::string, std::string> d = makeCacheNames(
    shapeOpt, nSamples, knn, alpha, noiseStd, nEv, "./cache/");

  std::cout << "Checking the dictionary---" << std::endl;
  for (auto& k : d){
    std::cout << k.first << ", " << k.second << std::endl;
  }

  std::cout << "\nTEST2---" << std::endl;
  std::map<std::string, std::string> cacheDic = makeCacheNames(
    shapeOpt, nSamples, knn, alpha, noiseStd, nEv, "./somedir/");
  const char* path_ls = cacheDic.at("ls").c_str();
  const char* path_lt = cacheDic.at("lt").c_str();
  const char* path_s_wl = cacheDic.at("s_wl").c_str();
  const char* path_t_wl = cacheDic.at("t_wl").c_str();
  const char* path_s_evals = cacheDic.at("s_evals").c_str();
  const char* path_t_evals = cacheDic.at("t_evals").c_str();

  std::cout << "checking---" << std::endl;
  std::cout << path_ls << std::endl;
  std::cout << path_lt << std::endl;
  std::cout << path_s_wl << std::endl;
  std::cout << path_t_wl << std::endl;
  std::cout << path_s_evals << std::endl;
  std::cout << path_t_evals << std::endl;
}

TEST(laplace_beltrami, f_height){
  ManagedHostImage<Vector3fda> pc(10);
  GetSimplePc(pc);
  printImage(pc,0,pc.Area());

  Eigen::VectorXf f_w(pc.Area());
  f_height(pc,f_w);
  std::cout << f_w.transpose() << std::endl;

  std::cout << "Test with sphere" << std::endl;
  GetSphericalPc(pc,10);
  printImage(pc, 0, pc.Area());

  f_height(pc,f_w);
  std::cout << f_w.transpose() << std::endl;

  std::cout << "finished." << std::endl;
}

TEST(laplace_beltrami, GetPc){
  std::pair<int, std::string> shapeOpt;
  int nSamples = 10;
  ManagedHostImage<Vector3fda> pc(nSamples,1);

  shapeOpt = std::make_pair(0,"");
  GetPc(pc, shapeOpt, nSamples);
  std::cout << "linear pc: ";
  printImage(pc,0,pc.Area());

  shapeOpt = std::make_pair(1,"");
  GetPc(pc, shapeOpt, nSamples);
  std::cout << "spherical pc: ";
  printImage(pc,0,pc.Area());

  shapeOpt = std::make_pair(2,"/home/hjsong/workspace/data/mesh/bun_zipper_res4.ply");
  GetPc(pc, shapeOpt, nSamples);
  std::cout << "bunny pc: ";
  printImage(pc,0,pc.Area());


  shapeOpt = std::make_pair(3,"/home/hjsong/workspace/data/mesh/cleanCylinder_0");
  GetPc(pc, shapeOpt, nSamples);
  std::cout << "manequine pc: ";
  printImage(pc,0,pc.Area());

}

TEST(laplace_beltrami, GetPointsOnSphere){
  ManagedHostImage<Vector3fda> pc(10,1),pc_cart(10,1);
  GetPointsOnSphere(pc);
  std::cout << "points on sphere with spherical coordinates:\n ";
  printImage(pc,0,pc.Area());

  std::cout << "to cartisean\n";
  toCartisean(pc,pc_cart);
  printImage(pc_cart,0,pc_cart.Area());
  std::cout <<"\tcheck if they are on the sphere---";
  for (int i=0; i<pc_cart.Area(); ++i){
    std::cout << pc_cart[i].norm() << ", ";
  }
  std::cout << std::endl;

}
TEST(laplace_beltrami, scale){
  ManagedHostImage<Vector3fda> src(10,1), dst(10,1);
  GetSphericalPc(src,10);
  std::cout << "src\n" << std::endl;
  printImage(src,0,src.Area());

  scale(src, 1.0f, dst);
  std::cout << "scale: 1.0f" << std::endl;
  printImage(dst,0,dst.Area());

  scale(src,2.0f, dst);
  std::cout << "scale: 2.0f" << std::endl;
  printImage(dst,0,dst.Area());
}

TEST(laplace_beltrami, deform){
  int n(5);
  ManagedHostImage<tdp::Vector3fda> pc(n,1),pc_cart(n,1),pc_d(n,1);
  GetPointsOnSphere(pc, n, 1);
  toCartisean(pc,pc_cart);

  std::cout << "Check if points are on unit sphere: \n";
  printImage(pc,0,pc.Area());
  std::cout << std::endl;
  // std::cout << "check norm: \n ";
  // tdp::printImage(pc_cart, 0, pc_cart.Area());
  // for(int i=0; i<pc_cart.Area(); ++i){
  //   std::cout << pc_cart[i].norm() << ", ";
  // }

  //Deformation
  float max_phi = M_PI_2;
  Deform(pc, pc_d, max_phi);
  std::cout << "Deformed---\n";
  std::cout << pc_d.Area() << std::endl;

  printImage(pc_d, 0, pc_d.Area());
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

  std::cout << "test laplace beltrami" << std::endl;
    return RUN_ALL_TESTS();
}

