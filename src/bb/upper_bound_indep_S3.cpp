/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <tdp/bb/upper_bound_indep_S3.h>

namespace tdp {

UpperBoundIndepS3::UpperBoundIndepS3(
      const std::vector<vMF3f>& vmf_mm_A, 
      const std::vector<vMF3f>& vmf_mm_B)
    : vmf_mm_A_(vmf_mm_A), vmf_mm_B_(vmf_mm_B)
{}

float UpperBoundIndepS3::Evaluate(const NodeS3& node) {
  std::vector<Eigen::Quaternion<float>> qs(4);
  for (uint32_t i=0; i<4; ++i)
    qs[i] = node.GetTetrahedron().GetVertexQuaternion(i);
  return EvaluateRotationSet(qs);
}

float UpperBoundIndepS3::EvaluateRotationSet(const
    std::vector<Eigen::Quaternion<float>>& qs) const {

  Eigen::VectorXf ubElem(vmf_mm_A_.size()*vmf_mm_B_.size());
  for (std::size_t j=0; j < vmf_mm_A_.size(); ++j)
    for (std::size_t k=0; k < vmf_mm_B_.size(); ++k) {
      Eigen::Vector3f p_star = ClosestPointInRotationSet(vmf_mm_A_[j],
          vmf_mm_B_[k], qs);
//      std::cout << "p_star " << p_star.transpose() << std::endl;
      ubElem(j*vmf_mm_B_.size() + k) =
        ComputeLogvMFtovMFcost<float,3>(vmf_mm_A_[j], vmf_mm_B_[k],
            p_star);
//      std::cout << "ubElem " << ubElem(j*vmf_mm_B_.size() + k) << std::endl;
    }
  return SumExp(ubElem);
}

float UpperBoundIndepS3::EvaluateAndSet(NodeS3& node) {
  float ub = Evaluate(node);
  node.SetUB(ub);
  return ub;
}

Eigen::Vector3f ComputeExtremumOnGeodesic(const Eigen::Vector3f& q1,
    const Eigen::Vector3f& q2, const Eigen::Vector3f& p, bool verbose) {
  const float theta12 = acos(std::min(1.f, std::max(-1.f, (q1.transpose()*q2)(0))));
  const float theta1 =  acos(std::min(1.f, std::max(-1.f, (q1.transpose()*p)(0))));
  const float theta2 =  acos(std::min(1.f, std::max(-1.f, (q2.transpose()*p)(0))));
  if (verbose)
    std::cout << "theta: " << theta12*180./M_PI << " "  << theta1*180./M_PI
      << " "  << theta2*180./M_PI << std::endl;
  float t = 0.5;
  float eps = 1.e-6/180.*M_PI;

  if (fabs(theta1-M_PI*0.5) < eps && fabs(theta2-M_PI*0.5) < eps) {
    if(verbose) std::cout << "  picking middle point. " << std::endl;
    t = 0.5;
  } else if (fabs(theta12) < eps) {
    if(verbose) std::cout << "  points are equal. " << std::endl;
    return (q1+q2)*0.5; // q1 \approx q2;
  }
  t = atan2(cos(theta2) - cos(theta12)*cos(theta1),
      cos(theta1)*sin(theta12)) / theta12;
  t = std::min(1.f, std::max(0.f, t));
  if(verbose) std::cout << "  on geodesic at " << t << std::endl;
  return (q1*sin((1.-t)*theta12) + q2*sin(t*theta12))/sin(theta12);
}

Eigen::Vector3f ClosestPointInRotationSet(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const std::vector<Eigen::Quaternion<float>>& qs, bool
    furthest, bool verbose) {
  Eigen::Vector3f muA = vmf_A.GetMu();
//  std::cout << " muA " << muA.transpose() << std::endl;
  if (furthest) muA *= -1.;
  std::vector<Eigen::Vector3f> mus(qs.size());
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
  Eigen::Matrix3f A;
  Combinations combinations(qs.size(),3);
  for (auto tri : combinations.Get()) {
    A << mus[tri[0]], mus[tri[1]], mus[tri[2]];
    // Check if muA inside triangle of rotated muBs
    Eigen::ColPivHouseholderQR<Eigen::Matrix3f> qr(A);
    if (qr.rank() == 3) {
      Eigen::Vector3f a = qr.solve(muA);
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
  Eigen::MatrixXf ps(3, qs.size()+(qs.size()*(qs.size()-1))/2);
  uint32_t k = 0;
  for (uint32_t i=0; i<qs.size(); ++i) {
    ps.col(k++) = mus[i];
    for (uint32_t j=i+1; j<qs.size(); ++j)
      ps.col(k++) = ComputeExtremumOnGeodesic(mus[i],
          mus[j], vmf_A.GetMu(), verbose);
  }
  Eigen::VectorXf dots = ps.transpose()*vmf_A.GetMu();
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

Eigen::Vector3f FurthestPointInRotationSet(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const std::vector<Eigen::Quaternion<float>>& qs, 
    bool verbose) {
  return ClosestPointInRotationSet(vmf_A, vmf_B, qs, true, verbose);
}

Eigen::Vector3f ClosestPointInTetrahedron(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const Tetrahedron4D& tetrahedron, bool furthest,
    bool verbose) {
  std::vector<Eigen::Quaternion<float>> qs(4);
  for (uint32_t i=0; i<4; ++i)
    qs[i] = tetrahedron.GetVertexQuaternion(i);
  return ClosestPointInRotationSet(vmf_A, vmf_B, qs, furthest, verbose);
}

Eigen::Vector3f FurthestPointInTetrahedron(const vMF3f& vmf_A, const
    vMF3f& vmf_B, const Tetrahedron4D& tetrahedron, bool verbose) {
  return ClosestPointInTetrahedron(vmf_A, vmf_B, tetrahedron, true, verbose);
}
}
