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

#include <tdp/gui/quickView.h>
#include <tdp/drivers/inertial/3dmgx3_45.h>
#include <tdp/manifold/SO3.h>
#include <tdp/manifold/SE3.h>

int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";
  if( argc > 1 ) {
    const std::string input_uri = std::string(argv[1]);
  }

  // Create OpenGL window - guess sensible dimensions
  int menue_w = 180;
  pangolin::CreateWindowAndBind( "GUI", 1000+menue_w, 800);

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

  tdp::Imu3DMGX3_45 imu("/dev/ttyACM0", 10);
  imu.Start();

  tdp::SO3f R_wi;
  tdp::ImuObs imuObsPrev;
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    // get next frames from the video source
    tdp::ImuObs imuObs;
    size_t numObs = 0;
    while(imu.GrabNext(imuObs)) {

      logAcc.Log(imuObs.acc(0),imuObs.acc(1),imuObs.acc(2));
      logMag.Log(imuObs.rpy(0), imuObs.rpy(1), imuObs.rpy(2));
      logOmega.Log(imuObs.omega(0), imuObs.omega(1), imuObs.omega(2));

      R_wi = R_wi.Exp(imuObs.omega*1e-9*(imuObs.t_host-imuObsPrev.t_host));
      imuObsPrev = imuObs;
      numObs ++;
    }

    plotAcc.ScrollView(numObs,0);
    plotMag.ScrollView(numObs,0);
    plotOmega.ScrollView(numObs,0);

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    // draw the axis
    tdp::SE3f T_wi(R_wi.matrix(), Eigen::Vector3f(0,0,0));
    pangolin::glDrawAxis<float>(T_wi.matrix(),1.1f);

    R_wi = tdp::SO3f::R_rpy(imuObs.rpy);
    T_wi = tdp::SE3f(R_wi.matrix(), Eigen::Vector3f(1,0,0));
    pangolin::glDrawAxis<float>(T_wi.matrix(),1.1f);
    glDisable(GL_DEPTH_TEST);
    // finish this frame
    pangolin::FinishFrame();
  }
  imu.Stop();

  return 0;
}
