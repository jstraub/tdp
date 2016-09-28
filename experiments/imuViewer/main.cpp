/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <Eigen/Dense>

#include <pangolin/pangolin.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/utils/timer.h>

#include <tdp/gui/quickView.h>
#include <tdp/drivers/inertial/3dmgx3_45.h>
#include <tdp/inertial/imu_outstream.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SE3.h>

#include <tdp/inertial/pose_interpolator.h>

int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  std::string input_uri = "";
  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
  }

  // Create OpenGL window - guess sensible dimensions
  int menue_w = 180;
  pangolin::CreateWindowAndBind( "GUI", 1000+menue_w, 800);
  pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menue_w));

  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
    .SetBounds(0., 1.0, pangolin::Attach::Pix(menue_w), 1.0);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(d_cam);

  pangolin::View& plotters = pangolin::Display("plotters");
  plotters.SetLayout(pangolin::LayoutEqualVertical);
  pangolin::DataLog logAcc;
  pangolin::Plotter plotAcc(&logAcc, -1000.f,1.f, -10.f,10.f, 10.f, 0.1f);
  plotters.AddDisplay(plotAcc);
  pangolin::DataLog logOmega;
  pangolin::Plotter plotOmega(&logOmega, -1000.f,1.f, -10.f,10.f, 10.f, 0.1f);
  plotters.AddDisplay(plotOmega);
  pangolin::DataLog logMag;
  pangolin::Plotter plotMag(&logMag, -1000.f,1.f, -10.f,10.f, 10.f, 0.1f);
  plotters.AddDisplay(plotMag);
  container.AddDisplay(plotters);

  pangolin::Var<bool> logImu("ui.log IMU", true, true);
  pangolin::Var<bool> verbose("ui.verbose", false, true);
  pangolin::Var<bool> applyCalib("ui.apply R_ir", false, true);
  pangolin::Var<bool> useTHost("ui.use t_host", false, true);

  tdp::Imu3DMGX3_45 imu("/dev/ttyACM0", 100);
  imu.Start();

  tdp::ImuOutStream imu_out("./testImu.pango");
  imu_out.Open(input_uri, imu.GetProperties());

  tdp::PoseInterpolator imuInterp;
  tdp::ThreadedValue<bool> receiveImu(true);
  tdp::ThreadedValue<size_t> numReceived(0);
  std::thread receiverThread (
    [&]() {
      tdp::ImuObs imuObs;
      tdp::ImuObs imuObsPrev;
      while(receiveImu.Get()) {
        if (imu.GrabNext(imuObs)) {

          Eigen::Matrix<float,6,1> se3 = Eigen::Matrix<float,6,1>::Zero();
          se3.topRows(3) = imuObs.omega;
          if (numReceived.Get() == 0) {
            imuInterp.Add(imuObs.t_host, tdp::SE3f());
          } else {
            int64_t dt_ns = imuObs.t_device - imuObsPrev.t_device;
            imuInterp.Add(imuObs.t_host, se3, dt_ns);
          }
          imuObsPrev = imuObs;
          numReceived.Increment();

          logAcc.Log(imuObs.acc(0),imuObs.acc(1),imuObs.acc(2));
          logMag.Log(imuObs.rpy(0), imuObs.rpy(1), imuObs.rpy(2));
          logOmega.Log(imuObs.omega(0), imuObs.omega(1), imuObs.omega(2));

          if (logImu)
            imu_out.WriteStream(imuObs);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });

  Eigen::Matrix3f R_ir;
  R_ir << 0, 0,-1,
       0,-1, 0,
       -1, 0, 0;
  tdp::SE3f T_ir(R_ir,Eigen::Vector3f::Zero());

  tdp::SO3f R_wi;
  tdp::ImuObs imuObsPrev;
  size_t numReceivedPrev = 0;
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source

    size_t numObs = numReceived.Get() - numReceivedPrev;
    plotAcc.ScrollView(numObs,0);
    plotMag.ScrollView(numObs,0);
    plotOmega.ScrollView(numObs,0);
    numReceivedPrev = numReceived.Get();

    int64_t tNow = pangolin::Time_us(pangolin::TimeNow())*1000;
    tdp::SE3f T_wi = imuInterp[tNow];

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    pangolin::glDrawAxis(1.0f);
    // draw the axis
    if (applyCalib) {
      T_wi = T_wi * T_ir;
    }
    pangolin::glDrawAxis<float>(T_wi.matrix(),0.8f);
    glDisable(GL_DEPTH_TEST);
    // finish this frame
    pangolin::FinishFrame();
  }
  receiveImu.Set(false);
  receiverThread.join();
  imu.Stop();
  imu_out.Close();

  return 0;
}
