/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

namespace tdp {

template<class NodeLin>
LowerBoundLin<NodeLin>::LowerBoundLin(LowerBoundS3& boundS3) 
  : boundS3_(boundS3)
{ }

template<class NodeLin>
float LowerBoundLin<NodeLin>::Evaluate(const NodeLin& node) {
  // We evaluate at the center of the center tetrahedron.
  // Only one evaluation point to be fair with direct S tessellation.
//  Eigen::Matrix<float,5,1> lbs;
//  for (uint32_t i=0; i<5; ++i)
//    lbs(i) = boundS3_.Evaluate(node.GetNodeS3(i));
//  return lbs.minCoeff();

  // Last node is the center tetrahedron according to
  // NodeLin::Linearize()
  Eigen::VectorXf lbs(1);
  std::vector<Eigen::Quaternion<float>> qs(1,node.GetCenter());
  boundS3_.EvaluateRotationSet(qs, lbs);
  return lbs(0);
}

template<class NodeLin>
float LowerBoundLin<NodeLin>::EvaluateAndSet(NodeLin& node) {
//  Eigen::Matrix<float,5,1> lbs;
//  uint32_t id = 0;
//  for (uint32_t i=0; i<5; ++i)
//    lbs(i) = boundS3_.Evaluate(node.GetNodeS3(i));
//  float lb = lbs.minCoeff(&id);
//  float lb = boundS3_.Evaluate(node.GetNodeS3(id));
//  Eigen::Matrix<float,3,9> xs;
//  Eigen::Matrix<float,9,1> lbs;
//  Evaluate(node, xs, lbs);
//  uint32_t id_max = 0;
////  float lb = lbs.maxCoeff(&id_max);
//  float lb = lbs(id_max);
//  node.SetLbArgument(xs.col(id_max));

  Eigen::VectorXf lbs(1);
  std::vector<Eigen::Quaternion<float>> qs(1,node.GetCenter());
  boundS3_.EvaluateRotationSet(qs, lbs);
  uint32_t id = 0;
  float lb = lbs(0);

  node.SetLB(lb);
  // Set the LB argument in the S3 node
//  boundS3_.EvaluateAndSet(node.GetNodeS3(id));
  // Copy the LB argument over to the Lin node
  node.SetLbArgument(qs[0]);
  return lb;
}

}
