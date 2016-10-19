#pragma once
#include <Eigen/Dense>
#include <tdp/data/pyramid.h>
#include <tdp/data/image.h>
#include <tdp/manifold/SL3.h>
#include <tdp/camera/homography.h>

namespace tdp {
class ESM {
 public:

  template<int LEVELS>
  static void EstimateHomography(
    const Pyramid<float,LEVELS>& gray_m,
    const Pyramid<float,LEVELS>& gradDu_m,
    const Pyramid<float,LEVELS>& gradDv_m,
    const Pyramid<float,LEVELS>& gray_c,
    const Pyramid<float,LEVELS>& gradDu_c,
    const Pyramid<float,LEVELS>& gradDv_c,
    SL3<float>& H);

  static void EstimateHomography(
    const Image<float>& gray_m,
    const Image<float>& gradDu_m,
    const Image<float>& gradDv_m,
    const Image<float>& gray_c,
    const Image<float>& gradDu_c,
    const Image<float>& gradDv_c,
    SL3<float>& H
    );

  static Eigen::Matrix<float,3,9> J_w(float u, float v) {
    Eigen::Matrix<float,3,9> J = Eigen::Matrix<float,3,9>::Zero();
    J(0,0) = u;
    J(0,1) = v;
    J(0,2) = 1;
    J(1,3) = u;
    J(1,4) = v;
    J(1,5) = 1;
    J(0,6) = -u*u;
    J(0,7) = -u*v;
    J(0,8) = -u;
    J(1,6) = -v*u;
    J(1,7) = -v*v;
    J(1,8) = -v;
    return J;
  }
};

template<int LEVELS>
void ESM::EstimateHomography(
    const Pyramid<float,LEVELS>& gray_m,
    const Pyramid<float,LEVELS>& gradDu_m,
    const Pyramid<float,LEVELS>& gradDv_m,
    const Pyramid<float,LEVELS>& gray_c,
    const Pyramid<float,LEVELS>& gradDu_c,
    const Pyramid<float,LEVELS>& gradDv_c,
    SL3<float>& H
    ) {
  for (int lvl = LEVELS-1; lvl >= 0; --lvl) {
    const Image<float> gray_ml = gray_m.GetConstImage(lvl);
    const Image<float> gradDu_ml = gradDu_m.GetConstImage(lvl);
    const Image<float> gradDv_ml = gradDv_m.GetConstImage(lvl);
    const Image<float> gray_cl = gray_c.GetConstImage(lvl);
    const Image<float> gradDu_cl = gradDu_c.GetConstImage(lvl);
    const Image<float> gradDv_cl = gradDv_c.GetConstImage(lvl);

    size_t maxIt = 10;
    for (size_t it=0; it<maxIt; ++it) {
      std::cout << "@" <<it << std::endl;
      ESM::EstimateHomography(
          gray_ml,
          gradDu_ml,
          gradDv_ml,
          gray_cl,
          gradDu_cl,
          gradDv_cl,
          H);
    }
    std::cout << H.matrix() << std::endl;
    std::cout << H.matrix().determinant() << std::endl;
  }
}

void ESM::EstimateHomography(
    const Image<float>& gray_m,
    const Image<float>& gradDu_m,
    const Image<float>& gradDv_m,
    const Image<float>& gray_c,
    const Image<float>& gradDu_c,
    const Image<float>& gradDv_c,
    SL3<float>& H
    ) {
  // precompute Jacobians
  Eigen::Matrix<float,9,8> J_g;
  for (size_t i=0; i<8; ++i) {
    for (size_t j=0; j<9; ++j) {
      J_g(j,i) = SL3<float>::G(i)(j/3,j%3);
    }
  }

  Eigen::Matrix<float,8,8> JTJ = Eigen::Matrix<float,8,8>::Zero();
  Eigen::Matrix<float,8,1> JTy = Eigen::Matrix<float,8,1>::Zero();
  float F = 0;

  Homography<float> Hcur(H.matrix());
  int w = gradDu_m.w_;
  int h = gradDu_m.h_;
  for (int u=1; u<w-1; ++u) {
    for (int v=1; v<h-1; ++v) {
      float x,y;
      Hcur.Transform(u,v,x,y);
//      std::cout << u << " " << v << " " 
//        << x << " " << y 
//        << " " << w << " " << h
//        << std::endl;
      if (x==x && y==y && 0.5 < x && x < w-1.5 && 0.5 < y && y < h-1.5) {
        Eigen::Matrix<float,1,3> J_c(gradDu_c.GetBilinear(x,y), 
            gradDv_c.GetBilinear(x,y), 0.);
        Eigen::Matrix<float,1,3> J_m(gradDu_m(u,v), gradDv_m(u,v), 0.);
        Eigen::Matrix<float,1,8> J_esm =  0.5*(J_c+J_m)*J_w(u,v)*J_g;
        float f = gray_c.GetBilinear(x,y) - gray_m(u,v);
        JTJ += J_esm.transpose() * J_esm;
        JTy += J_esm.transpose() * f;
        F += f;
//        std::cout << f << ": " << J_esm << std::endl;
      }
    }
  }

  Eigen::Matrix<double,8,1> x = JTJ.cast<double>().ldlt().solve(JTy.cast<double>());
  std::cout << F << ": " << x.transpose() << std::endl;
//  std::cout << JTJ << std::endl << JTy.transpose() << std::endl;
//  std::cout << H.matrix() << std::endl;
  H = H * SL3<float>(SL3f::Exp_(x.cast<float>()));
//  std::cout << H.matrix() << std::endl;
//  std::cout << H.matrix().determinant() << std::endl;

}


}
