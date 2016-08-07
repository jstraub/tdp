#include <iostream>
#include <Eigen/Dense>
#include <manifold/S.h>

int main (int argc, char** argv) {
  
  S3d q;
  std::cout << q << std::endl;

  std::cout << q.Exp(Eigen::Vector3d(0.,M_PI/2.,0.)).norm() << std::endl;
  std::cout << q.Exp(q.ToAmbient(Eigen::Vector2d(0.,M_PI/2.))).norm() << std::endl;

  S3d mu;
  mu.vector() << 1./sqrt(2),1./sqrt(2), 0.;
  std::cout << mu << std::endl;

  double delta = 0.1;
  double f_prev = 1e99;
  double f = mu.dot(q);
  std::cout << "f=" << f << std::endl;
  for (uint32_t it=0; it<100; ++it) {
    Eigen::Vector3d J = -2.*(mu.vector() - q.vector()*q.dot(mu)); 

    q = q.Exp(-delta*J);
//    q = q.RetractOrtho(-delta*J);

    f_prev = f;
    f = mu.dot(q);
    std::cout << "@" << it << ": f=" << f 
      << " df/f=" << (f_prev - f)/fabs(f) << std::endl;
    if ((f_prev - f)/fabs(f) > -1e-10) break;
  }
  std::cout << std::endl << mu << std::endl;
  std::cout << std::endl << q << std::endl;
}

