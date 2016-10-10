/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iomanip>
#include <tdp/inertial/imu_interpolator.h>

namespace tdp {

void ImuInterpolator::Start() {
  receiveImu_.Set(true);
  receiverThread_ = std::thread([&]() {
      tdp::ImuObs imuObs;
      tdp::ImuObs imuObsPrev;
      int numCalib = 0;
      while(receiveImu_.Get()) {
        if (imu_ && imu_->GrabNext(imuObs)) {
          if (out_ && record_.Get()) out_->WriteStream(imuObs);

          if (!calibrated_ && numReceived_.Get() > 10 
            && imuObs.omega.norm() < 2./180.*M_PI) {
            gyro_bias_ += imuObs.omega;
            gravity0_ += imuObs.acc;
            numCalib ++;
            std::cout << "rotVel: " 
              << std::setprecision(3) << imuObs.omega.norm()*180./M_PI 
              << "\tacc: " 
              << std::setprecision(3) << imuObs.acc.norm() << std::endl;
          } else if (!calibrated_ && numReceived_.Get() > 10) {
            calibrated_ = true;
            gyro_bias_ /= numCalib;
            gravity0_ /= numCalib;
            std::cout << "IMU calibrated. gyro " << gyro_bias_.transpose() 
              << " gravity: " << gravity0_.transpose()
              << std::endl;
          } else {
            imuObs.omega -= gyro_bias_;
          }

          Eigen::Matrix<float,6,1> se3 = Eigen::Matrix<float,6,1>::Zero();
          se3.topRows(3) = imuObs.omega;
          if (numReceived_.Get() == 0) {
            Ts_wi_.Add(imuObs.t_host, tdp::SE3f());
          } else {
            int64_t dt_ns = imuObs.t_device - imuObsPrev.t_device;
            Ts_wi_.Add(imuObs.t_host, se3, dt_ns);
          }
          imuObsPrev = imuObs;
          numReceived_.Increment();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });
}
void ImuInterpolator::Stop() {
  receiveImu_.Set(false);
  receiverThread_.join();
}

}
