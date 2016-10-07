/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/upper_bound_indep_S3.h>

namespace tdp {

UpperBoundIndepS3::UpperBoundIndepS3(const vMFMM<3>& vmf_mm_A, const vMFMM<3>&
    vmf_mm_B) : vmf_mm_A_(vmf_mm_A), vmf_mm_B_(vmf_mm_B)
{}

double UpperBoundIndepS3::Evaluate(const NodeS3& node) {
  std::vector<Eigen::Quaterniond> qs(4);
  for (uint32_t i=0; i<4; ++i)
    qs[i] = node.GetTetrahedron().GetVertexQuaternion(i);
  return EvaluateRotationSet(qs);
}

double UpperBoundIndepS3::EvaluateRotationSet(const
    std::vector<Eigen::Quaterniond>& qs) const {

  Eigen::VectorXd ubElem(vmf_mm_A_.GetK()*vmf_mm_B_.GetK());
  for (std::size_t j=0; j < vmf_mm_A_.GetK(); ++j)
    for (std::size_t k=0; k < vmf_mm_B_.GetK(); ++k) {
      Eigen::Vector3d p_star = ClosestPointInRotationSet(vmf_mm_A_.Get(j),
          vmf_mm_B_.Get(k), qs);
//      std::cout << "p_star " << p_star.transpose() << std::endl;
      ubElem(j*vmf_mm_B_.GetK() + k) =
        ComputeLogvMFtovMFcost<3>(vmf_mm_A_.Get(j), vmf_mm_B_.Get(k),
            p_star);
//      std::cout << "ubElem " << ubElem(j*vmf_mm_B_.GetK() + k) << std::endl;
    }
  return SumExp(ubElem);
}

double UpperBoundIndepS3::EvaluateAndSet(NodeS3& node) {
  double ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

Eigen::Vector3d ComputeExtremumOnGeodesic(const Eigen::Vector3d& q1,
    const Eigen::Vector3d& q2, const Eigen::Vector3d& p, bool verbose) {
  const double theta12 = acos(std::min(1., std::max(-1., (q1.transpose()*q2)(0))));
  const double theta1 = acos(std::min(1., std::max(-1., (q1.transpose()*p)(0))));
  const double theta2 = acos(std::min(1., std::max(-1., (q2.transpose()*p)(0))));
  if (verbose)
    std::cout << "theta: " << theta12*180./M_PI << " "  << theta1*180./M_PI
      << " "  << theta2*180./M_PI << std::endl;
  double t = 0.5;
  double eps = 1.e-6/180.*M_PI;

  if (fabs(theta1-M_PI*0.5) < eps && fabs(theta2-M_PI*0.5) < eps) {
    if(verbose) std::cout << "  picking middle point. " << std::endl;
    t = 0.5;
  } else if (fabs(theta12) < eps) {
    if(verbose) std::cout << "  points are equal. " << std::endl;
    return (q1+q2)*0.5; // q1 \approx q2;
  }
  t = atan2(cos(theta2) - cos(theta12)*cos(theta1),
      cos(theta1)*sin(theta12)) / theta12;
  t = std::min(1., std::max(0., t));
  if(verbose) std::cout << "  on geodesic at " << t << std::endl;
  return (q1*sin((1.-t)*theta12) + q2*sin(t*theta12))/sin(theta12);
}

Eigen::Vector3d ClosestPointInRotationSet(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const std::vector<Eigen::Quaterniond>& qs, bool
    furthest, bool verbose) {
  Eigen::Vector3d muA = vmf_A.GetMu();
//  std::cout << " muA " << muA.transpose() << std::endl;
  if (furthest) muA *= -1.;
  std::vector<Eigen::Vector3d> mus(qs.size());
  for (uint32_t i=0; i<qs.size(); ++i) {
    mus[i] = qs[i]._transformVector(vmf_B.GetMu());
//    std::cout << " muB " << i << " " 
//      << tetrahedron.GetVertex(i).transpose() << " -> "
//      << tetrahedron.GetVertexQuaternion(i).coeffs().transpose() << " -> "
//      << mus[i].transpose() << std::endl;
  }
  if(verbose) {
    std::cout << "-- Polygone:\n";
    for (auto& mu : mus)
      std::cout << mu.transpose() << std::endl;
    std::cout << " query:\n" << vmf_A.GetMu().transpose(); 
  }
  // Check if muA is in any of the triangles spanned by the rotated
  // muBs
  Eigen::Matrix3d A;
  Combinations combinations(qs.size(),3);
  for (auto tri : combinations.Get()) {
    A << mus[tri[0]], mus[tri[1]], mus[tri[2]];
    // Check if muA inside triangle of rotated muBs
    Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr(A);
    if (qr.rank() == 3) {
      Eigen::Vector3d a = qr.solve(muA);
      if ((a.array() > 0.).all()) {
        if(verbose) {
          if (furthest)
            std::cout << " furthest point inside polygone " <<
              muA.transpose() << std::endl;
          else 
            std::cout << " closest point inside polygone " <<
              muA.transpose() << std::endl;
        }
        return muA;
      }
    }
  }
  // Check the edges and corners.
  Eigen::MatrixXd ps(3, qs.size()+(qs.size()*(qs.size()-1))/2);
  uint32_t k = 0;
  for (uint32_t i=0; i<qs.size(); ++i) {
    ps.col(k++) = mus[i];
    for (uint32_t j=i+1; j<qs.size(); ++j)
      ps.col(k++) = ComputeExtremumOnGeodesic(mus[i],
          mus[j], vmf_A.GetMu(), verbose);
  }
  Eigen::VectorXd dots = ps.transpose()*vmf_A.GetMu();
//  std::cout << "dots " << dots.transpose() << std::endl;
  uint32_t id = 0;
  if (furthest) 
    dots.minCoeff(&id);
  else
    dots.maxCoeff(&id);

  if (verbose) {
    if (furthest) {
      std::cout << " furthest point on polygone:\n" << ps.col(id).transpose() 
        << std::endl;
    } else {
      std::cout << " closest point on polygone:\n" << ps.col(id).transpose() 
        << std::endl;
    }
  }
  return ps.col(id);
}

Eigen::Vector3d FurthestPointInRotationSet(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const std::vector<Eigen::Quaterniond>& qs, 
    bool verbose) {
  return ClosestPointInRotationSet(vmf_A, vmf_B, qs, true, verbose);
}

Eigen::Vector3d ClosestPointInTetrahedron(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const Tetrahedron4D& tetrahedron, bool furthest,
    bool verbose) {
  std::vector<Eigen::Quaterniond> qs(4);
  for (uint32_t i=0; i<4; ++i)
    qs[i] = tetrahedron.GetVertexQuaternion(i);
  return ClosestPointInRotationSet(vmf_A, vmf_B, qs, furthest, verbose);
}

Eigen::Vector3d FurthestPointInTetrahedron(const vMF<3>& vmf_A, const
    vMF<3>& vmf_B, const Tetrahedron4D& tetrahedron, bool verbose) {
  return ClosestPointInTetrahedron(vmf_A, vmf_B, tetrahedron, true, verbose);
}
}
