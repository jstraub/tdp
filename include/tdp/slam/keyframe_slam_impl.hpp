#include <tdp/slam/keyframe_slam.h>

namespace tdp {

KeyframeSLAM::KeyframeSLAM() :
  noisePrior_(isam::Information(10000. * isam::eye(6))),
  noiseOdom_(isam::Information(100. * isam::eye(6))),
  noiseLoopClosure_(isam::Information(100. * isam::eye(6)))
{ 
  isam::Properties props;
  props.verbose=false;
  props.quiet=true;
  props.method=isam::Method::DOG_LEG;
//  props.method=isam::Method::LEVENBERG_MARQUARDT;
  //props.method=isam::Method::GAUSS_NEWTON;
  props.epsilon_rel = 1e-8;
  props.epsilon_abs = 1e-6;
  props.mod_batch = 10;
  slam_.set_properties(props); 
};

KeyframeSLAM::~KeyframeSLAM()
{};

void KeyframeSLAM::AddOrigin(const SE3f& T_wk) {

  isam::Pose3d origin = isam::Pose3d(T_wk.matrix().cast<double>());

  isam::Pose3d_Node* pose0 = new isam::Pose3d_Node();
  slam_.add_node(pose0);
  T_wk_.push_back(pose0);

  isam::Pose3d_Factor* prior = new isam::Pose3d_Factor(pose0,
      origin, noisePrior_);
  slam_.add_factor(prior);
}

void KeyframeSLAM::AddPose() {
  isam::Pose3d_Node* poseNext = new isam::Pose3d_Node();
  slam_.add_node(poseNext);
  T_wk_.push_back(poseNext);
}

void KeyframeSLAM::AddIcpOdometry(int idA, int idB, const SE3f& T_ab) {

  isam::Pose3d poseAB(T_ab.matrix().cast<double>());

  isam::Pose3d_Node* poseNext = new isam::Pose3d_Node();
  slam_.add_node(poseNext);
  T_wk_.push_back(poseNext);

  isam::Pose3d_Pose3d_Factor* odo = new isam::Pose3d_Pose3d_Factor(
      T_wk_[idA], T_wk_[idB], poseAB, noiseOdom_);
  slam_.add_factor(odo);
  loopClosures_.emplace_back(idA, idB);
}

void KeyframeSLAM::AddLoopClosure(int idA, int idB, const SE3f& T_ab,
    const Eigen::Matrix<float,6,6>& Sigma_ab) {

  isam::Covariance noise(Sigma_ab.cast<double>());
  isam::Pose3d poseAB(T_ab.matrix().cast<double>());

  isam::Pose3d_Pose3d_Factor* loop = new isam::Pose3d_Pose3d_Factor(
      T_wk_[idA], T_wk_[idB], poseAB, noise);
  slam_.add_factor(loop);

  loopClosures_.emplace_back(idA, idB);
}

void KeyframeSLAM::AddLoopClosure(int idA, int idB, const SE3f& T_ab) {

  isam::Pose3d poseAB(T_ab.matrix().cast<double>());

  isam::Pose3d_Pose3d_Factor* loop = new isam::Pose3d_Pose3d_Factor(
      T_wk_[idA], T_wk_[idB], poseAB, noiseLoopClosure_);
  slam_.add_factor(loop);

  loopClosures_.emplace_back(idA, idB);
}

void KeyframeSLAM::PrintValues() {
  for (auto& T_wk : T_wk_) {
    std::cout << T_wk->value() << std::endl;
  }
}

void KeyframeSLAM::PrintGraph() {
  slam_.print_graph();
}

void KeyframeSLAM::Optimize() {
//    std::cout << "initial"<< std::endl;
//    PrintValues();
  std::cout << "SAM optimization" << std::endl;
//  slam_.batch_optimization();
  std::cout 
    << slam_.weighted_errors().sum()
    << " " << slam_.chi2()
    << std::endl;
  slam_.update();
  std::cout 
    << slam_.weighted_errors().sum()
    << " " << slam_.chi2()
    << std::endl;


//    slam_.print_stats();
  slam_.save("isam.csv");
//    slam_.print_graph();
//    std::cout << "after iterations" << std::endl;
//    PrintValues();
}

SE3f KeyframeSLAM::GetPose(size_t i) {
  if (i<size()) {
    Eigen::Matrix<float,4,4> T = T_wk_[i]->value().wTo().cast<float>();
    return SE3f(T);
  } 
  return SE3f(); 
}

}



