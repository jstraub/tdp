/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<typename T, class NodeLin>
LowerBoundLin<T,NodeLin>::LowerBoundLin(LowerBoundS3<T>& boundS3) 
  : boundS3_(boundS3)
{ }

template<typename T, class NodeLin>
T LowerBoundLin<T,NodeLin>::Evaluate(const NodeLin& node) {
  // We evaluate at the center of the center tetrahedron.
  // Only one evaluation point to be fair with direct S tessellation.
//  Eigen::Matrix<T,5,1> lbs;
//  for (uint32_t i=0; i<5; ++i)
//    lbs(i) = boundS3_.Evaluate(node.GetNodeS3(i));
//  return lbs.minCoeff();

  // Last node is the center tetrahedron according to
  // NodeLin::Linearize()
  Eigen::Matrix<T,Eigen::Dynamic,1> lbs(1);
  std::vector<Eigen::Quaternion<T>> qs(1,node.GetCenter());
  boundS3_.EvaluateRotationSet(qs, lbs);
  return lbs(0);
}

template<typename T, class NodeLin>
T LowerBoundLin<T,NodeLin>::EvaluateAndSet(NodeLin& node) {
//  Eigen::Matrix<T,5,1> lbs;
//  uint32_t id = 0;
//  for (uint32_t i=0; i<5; ++i)
//    lbs(i) = boundS3_.Evaluate(node.GetNodeS3(i));
//  T lb = lbs.minCoeff(&id);
//  T lb = boundS3_.Evaluate(node.GetNodeS3(id));
//  Eigen::Matrix<T,3,9> xs;
//  Eigen::Matrix<T,9,1> lbs;
//  Evaluate(node, xs, lbs);
//  uint32_t id_max = 0;
////  T lb = lbs.maxCoeff(&id_max);
//  T lb = lbs(id_max);
//  node.SetLbArgument(xs.col(id_max));

  Eigen::Matrix<T,Eigen::Dynamic,1> lbs(1);
  std::vector<Eigen::Quaternion<T>> qs(1,node.GetCenter());
  boundS3_.EvaluateRotationSet(qs, lbs);
  uint32_t id = 0;
  T lb = lbs(0);

  node.SetLB(lb);
  // Set the LB argument in the S3 node
//  boundS3_.EvaluateAndSet(node.GetNodeS3(id));
  // Copy the LB argument over to the Lin node
  node.SetLbArgument(qs[0]);
  return lb;
}

}
