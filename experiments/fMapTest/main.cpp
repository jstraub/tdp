#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>
#include <random>

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
#include <Eigen/Core>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/gl_draw.h>

#include <tdp/gui/gui.hpp>
#include <tdp/gui/quickView.h>

#include <tdp/nn/ann.h>
#include <tdp/manifold/S.h>
#include <tdp/manifold/SE3.h>
#include <tdp/data/managed_image.h>

#include <tdp/utils/status.h>
#include <tdp/utils/timer.hpp>
#include <tdp/eigen/std_vector.h>

#include <tdp/laplace_beltrami/laplace_beltrami.h>

/************Declarations***************************************
 ***************************************************************/
void printImage(const tdp::ManagedHostImage<tdp::Vector3fda>& pc,
                int start_idx,
                const int length);

void Test_printImage();
void Test_f_landmark();
void Test_projections();
void Test_samePc_exactPairs(std::string& option);

/************end delcarations************************************/


int main(){
    //Test_printImage();
    std::string option("rbf");
    Test_samePc_exactPairs(option);
}



/************Implementations***************************************
 ******************************************************************/
void printImage(const tdp::ManagedHostImage<tdp::Vector3fda>& pc,
                int start_idx,
                const int length){
    // prints pc[i] for i in [start_idx,start_idx + length -1 ]
    // pc[i] element's are comman separated
    // and printed out as transposed. 

    int end_idx = start_idx + length -1 ;
    if (start_idx < 0){
        start_idx = 0;
    }
    if (end_idx > pc.Area()-1){
        end_idx = pc.Area()-1;
    }
    //assert(start_idx >= 0 && end_idx < pc.Area());
    for (int i=start_idx; i<= end_idx; ++i){
        std::cout << pc[i].transpose() << ", ";
    }
    std::cout << std::endl;
}

void Test_printImage(){
    tdp::ManagedHostImage<tdp::Vector3fda> pc = tdp::GetSimplePc();
    std::cout << "---Printing image---" << std::endl;
    printImage(pc,0,pc.Area());

    pc.Reinitialise(10);
    for (int i =0; i< 10; ++i){
        pc[i] = tdp::Vector3fda(i,i,i);
    }
    std::cout << "---Printing image---" << std::endl;
    printImage(pc,0,pc.Area());
}

void Test_f_landmark(){
    std::string opt("rbf");
    tdp::ManagedHostImage<tdp::Vector3fda> pc = tdp::GetSimplePc();

}

void Test_projections(){
    // Test for projectToLocal and projectToWorld


}

void Test_samePc_exactPairs(std::string& option){//todo: std::option
    // parameters
    int nSamples = 1000;
    tdp::ManagedHostImage<tdp::Vector3fda> pc_s;
    tdp::ManagedHostImage<tdp::Vector3fda> pc_t;

    // Get random points from a sphere
    std::srand(101);
    tdp::GetSphericalPc(pc_s, nSamples);
    pc_t.ResizeCopyFrom(pc_s);

    int numEv = pc_s.Area()/2;//pc_s.Area()-2; //get ALL eigenvectors of L
    int knn = pc_s.Area(); // use all points as neighbors
    float eps = 1e-6;
    float alpha = 0.01;
    float alpha2 = 0.1;

    int numPW = numEv;//number of pointwise correspondences
    int numHKS = 0; //number of heat kernel signature correspondences
    int numCst = numPW + numHKS;//pc_s.Area();
    
    //int numTest = pc_s.Area() - numPW;

    // build kd tree
    tdp::ANN ann_s, ann_t;
    ann_s.ComputeKDtree(pc_s);
    ann_t.ComputeKDtree(pc_t);

    // construct laplacian matrices
    Eigen::SparseMatrix<float> L_s(pc_s.Area(), pc_s.Area()),
                               L_t(pc_t.Area(), pc_t.Area());
    Eigen::MatrixXf S_wl(L_s.rows(),(int)numEv),//cols are evectors
                    T_wl(L_t.rows(),(int)numEv),
                    S_desc_w, T_desc_w,
                    S_desc_l, T_desc_l;
    Eigen::VectorXf S_evals((int)numEv), T_evals((int)numEv);


    L_s = tdp::getLaplacian(pc_s, ann_s, knn, eps, alpha);
    L_t = tdp::getLaplacian(pc_t, ann_t, knn, eps, alpha);
    tdp::decomposeLaplacian(L_s, numEv, S_evals, S_wl); //todo: check if size initialization is included
    tdp::decomposeLaplacian(L_t, numEv, T_evals, T_wl);

    std::cout << "Basis ---" << std::endl;
    std::cout << S_wl << std::endl;
    std::cout << "-----------------" << std::endl;
    std::cout << T_wl << std::endl;
    std::cout << "Evals ---" << std::endl;
    std::cout << S_evals.transpose() << std::endl;
    std::cout << T_evals.transpose() << std::endl;


    //--Construct function pairs
    Eigen::VectorXf f_w(pc_s.Area()), g_w(pc_t.Area()),
                    f_l((int)numEv), g_l((int)numEv);
    Eigen::MatrixXf F((int)numCst, (int)numEv), G((int)numCst, (int)numEv);
    Eigen::MatrixXf C((int)numEv, (int)numEv);

    // --construct F(data matrix) and G based on the correspondences

    for (int i=0; i< (int)numPW; ++i){
        if (option == "rbf"){
            tdp::f_rbf(pc_s, pc_s[i], alpha2, f_w); //todo: check if I can use this same alpha?
            tdp::f_rbf(pc_t, pc_t[i], alpha2, g_w);
        } else {
            tdp::f_indicator(pc_s, i, f_w); //todo: check if I can use this same alpha?
            tdp::f_indicator(pc_t, i, g_w);
        }
        f_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*f_w);
        g_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*g_w);
        //f_l = tdp::projectToLocal(S_wl, f_w);
        //g_l = tdp::projectToLocal(T_wl, g_w);

        F.row(i) = f_l;
        G.row(i) = g_l;
    }


    if (numHKS >0){
        //-----Add  heat kernel signatures as constraints
        std::cout << "CALCULATEING HKS ---" <<std::endl;
        S_desc_w = tdp::getHKS(S_wl,S_evals,numHKS);
        T_desc_w = tdp::getHKS(T_wl,T_evals,numHKS);
        S_desc_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*S_desc_w);
        T_desc_l = (T_wl.transpose()*T_wl).fullPivLu().solve(T_wl.transpose()*T_desc_w);
        //S_desc_l = tdp::projectToLocal(S_wl, S_desc_w); //columne is a feature
        //T_desc_l = tdp::projectToLocal(T_wl, T_desc_w);
        
        assert(S_desc_l.cols() == numHKS);
        for (int i=0; i<numHKS; ++i){
          F.row(numPW+i) = S_desc_l.col(i);
          G.row(numPW+i) = T_desc_l.col(i);
        }
        std::cout << "S,T descriptors at time 0--------" << std::endl;
        std::cout << S_desc_l.col(0) << std::endl;//heat kernel at timestap i//todo: check at which point for S and T manifolds
        std::cout << T_desc_l.col(0) << std::endl;//heat kernel at timestap i
    }
    //----Add operator constratins
    //

    // solve least-square
    C = (F.transpose()*F).fullPivLu().solve(F.transpose()*G);
    //std::cout << "F: \n" << F.rows() << F.cols() << std::endl;
    //std::cout << "\nG: \n" << G.rows() << G.cols() << std::endl;
    std::cout << "\nC: \n" << C << /*C.rows() << C.cols() <<*/ std::endl;

    // Test
    assert(numPW < pc_s.Area());
    int numTest = (int)pc_s.Area()-numPW;
    float error = 0;
    for (int i=numPW; i< (int)pc_s.Area(); ++i ){
        tdp::Vector3fda true_w = pc_s[i];
        tdp::Vector3fda true_l = (S_wl.transpose()*S_wl).fullPivLu().solve(S_wl.transpose()*true_w);
        tdp::Vector3fda guess_w = S_wl * (C*true_l);
        // tdp::Vector3fda true_l = tdp::projectToLocal(S_wl, true_w);
        // tdp::Vector3fda guess_w = tdp::projectToWorld(S_wl, C*true_l);
        error += (true_w - guess_w).squaredNorm();
    }
    error /= numTest;
    std::cout << "Number of test points: " << numTest << std::endl;
    std::cout << "error: " << std::endl;

    // //Get correspondences
    // Eigen::VectorXi nnIds(1);
    // Eigen::VectorXf dists(1);
    // tdp::ManagedHostImage<tdp::Vector3fda> queries((int)numQ,1);
    // tdp::ManagedHostImage<tdp::Vector3fda> estimates((int)numQ,1);
    // for (int i=0; i<(int)numQ; ++i){
    //     int tId = getCorrespondence(pc_s, pc_t, S_wl, T_wl, C, alpha2, i); //guessed id in second manifold
    //     ann_t.Search(pc_s[i], 1, 1e-9, nnIds, dists);
    //     queries[i] = pc_s[i];
    //     estimates[i] = pc_t[tId];
    //     std::cout << "query: \n" << pc_s[i].transpose()<<std::endl;
    //     std::cout << "guess: \n" << pc_t[tId].transpose() << std::endl;
    //     std::cout << "true: \n" << pc_t[nnIds(0)].transpose() << std::endl;
    // }
}

