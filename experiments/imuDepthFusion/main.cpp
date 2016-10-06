/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <thread>
#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/gldraw.h>

#include <Eigen/Dense>
#include <tdp/camera/rig.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/data/managed_pyramid.h>
#include <tdp/data/managed_volume.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/volume.h>
#include <tdp/gui/gui.hpp>
#include <tdp/gui/quickView.h>
#include <tdp/icp/icp.h>
#include <tdp/manifold/SE3.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/preproc/convolutionSeparable.h>
#include <tdp/preproc/depth.h>
#include <tdp/preproc/normals.h>
#include <tdp/preproc/pc.h>
#include <tdp/tsdf/tsdf.h>
#include <tdp/utils/Stopwatch.h>
#include <tdp/inertial/imu_factory.h>
#include <tdp/inertial/imu_interpolator.h>
#include <tdp/manifold/SO3.h>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

using namespace gtsam;
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

typedef tdp::CameraPoly3f CameraT;
//typedef tdp::Cameraf CameraT;

int main( int argc, char* argv[] )
{
  std::string input_uri = "openni2://";
  std::string output_uri = "pango://video.pango";
  std::string calibPath = "";
  std::string imu_input_uri = "";
  std::string tsdfOutputPath = "tsdf.raw";

  if( argc > 1 ) {
    input_uri = std::string(argv[1]);
    calibPath = (argc > 2) ? std::string(argv[2]) : "";
    imu_input_uri =  (argc > 3)? std::string(argv[3]) : "";
  }

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return 1;
  }

  // optionally connect to IMU if it is found.
  tdp::ImuInterface* imu = nullptr; 
  if (imu_input_uri.size() > 0) 
    imu = tdp::OpenImu(imu_input_uri);
  if (imu) imu->Start();

  Eigen::Matrix3f R_ir;
  R_ir << 0, 0,-1,
       0,-1, 0,
       -1, 0, 0;
  tdp::SE3f T_ir(R_ir,Eigen::Vector3f::Zero());

  tdp::GUI gui(1200,800,video);
  size_t w = video.Streams()[gui.iRGB[0]].Width();
  size_t h = video.Streams()[gui.iRGB[0]].Height();
  size_t wc = w+w%64; // for convolution
  size_t hc = h+h%64;
  float f = 550;
  float uc = (w-1.)/2.;
  float vc = (h-1.)/2.;

  size_t dTSDF = 512;
  size_t wTSDF = 512;
  size_t hTSDF = 512;

  CameraT camR(Eigen::Vector4f(f,f,uc,vc)); 
  CameraT camD(Eigen::Vector4f(f,f,uc,vc)); 

  if (calibPath.size() > 0) {
    tdp::Rig<CameraT> rig;
    rig.FromFile(calibPath,false);
    std::vector<int32_t> rgbStream2cam;
    std::vector<int32_t> dStream2cam;
    std::vector<int32_t> rgbdStream2cam;
    std::vector<pangolin::VideoInterface*>& streams = video.InputStreams();
    tdp::CorrespondOpenniStreams2Cams(streams,rig,rgbStream2cam,
        dStream2cam, rgbdStream2cam);
    // camera model for computing point cloud and normals
    camR = rig.cams_[rgbStream2cam[0]];
    camD = camR; //rig.cams_[dStream2cam[0]];
    if (rig.T_ris_.size() > 0) T_ir = rig.T_ris_[0];
  }

  tdp::Camera<float> camView(Eigen::Vector4f(220,220,319.5,239.5)); 
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,319.5,239.5,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewPc3D);
  pangolin::View& viewN3D = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(viewN3D);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<Eigen::Matrix<uint8_t,3,1>> n2D(wc,hc);
  memset(n2D.ptr_,0,n2D.SizeBytes());
  tdp::ManagedHostImage<Eigen::Vector3f> n2Df(wc,hc);
  tdp::ManagedHostImage<Eigen::Vector3f> n(wc,hc);

  tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
  tdp::ManagedDeviceImage<float> cuD(wc, hc);

  tdp::ManagedHostVolume<tdp::TSDFval> TSDF(wTSDF, hTSDF, dTSDF);
  TSDF.Fill(tdp::TSDFval(-1.01,0.));
  tdp::ManagedDeviceVolume<tdp::TSDFval> cuTSDF(wTSDF, hTSDF, dTSDF);
  tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);

  tdp::ManagedHostImage<float> dEst(wc, hc);
  tdp::ManagedDeviceImage<float> cuDEst(wc, hc);
  dEst.Fill(0.);
  tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
  tdp::ManagedDeviceImage<float> cuDView(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcView(wc, hc);

  // ICP stuff
  tdp::ManagedHostPyramid<float,3> dPyr(wc,hc);
  tdp::ManagedHostPyramid<float,3> dPyrEst(wc,hc);
  tdp::ManagedDevicePyramid<float,3> cuDPyr(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
  tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);

  tdp::ManagedDeviceImage<float> cuPcErr(wc, hc);
  tdp::ManagedDeviceImage<float> cuAngErr(wc, hc);
  tdp::ManagedHostImage<float> pcErr(wc, hc);
  tdp::ManagedHostImage<float> angErr(wc, hc);

  tdp::ManagedDeviceImage<tdp::Vector3fda> nEstdummy(wc,hc);

  pangolin::GlBufferCudaPtr cuPcbuf(pangolin::GlArrayBuffer, wc*hc,
      GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

  tdp::ManagedHostImage<float> tsdfDEst(wc, hc);
  tdp::ManagedHostImage<float> tsdfSlice(wTSDF, hTSDF);
  tdp::QuickView viewTsdfDEst(wc,hc);
  tdp::QuickView viewTsdfSliveView(wTSDF,hTSDF);
  gui.container().AddDisplay(viewTsdfDEst);
  gui.container().AddDisplay(viewTsdfSliveView);

  tdp::ManagedHostImage<float> dispDepthPyr(dPyr.Width(0)+dPyr.Width(1), hc);
  tdp::QuickView viewDepthPyr(dispDepthPyr.w_,dispDepthPyr.h_);
  gui.container().AddDisplay(viewDepthPyr);
  
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuDispNormalsPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedDeviceImage<tdp::Vector3bda> cuDispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);
  tdp::ManagedHostImage<tdp::Vector3bda> dispNormals2dPyr(ns_m.Width(0)+ns_m.Width(1), hc);

  tdp::QuickView viewNormalsPyr(dispNormals2dPyr.w_,dispNormals2dPyr.h_);
  gui.container().AddDisplay(viewNormalsPyr);

  pangolin::Var<float> depthSensorScale("ui.depth sensor scale",1e-3,1e-4,1e-3);
  pangolin::Var<float> tsdfDmin("ui.d min",0.10,0.0,0.1);
  pangolin::Var<float> tsdfDmax("ui.d max",4.,0.1,4.);

  pangolin::Var<bool> dispNormalsPyrEst("ui.disp normal est", false, true);
  pangolin::Var<bool> runFusion("ui.run Fusion",true,true);

  pangolin::Var<bool>  resetTSDF("ui.reset TSDF", false, false);
  pangolin::Var<bool>  saveTSDF("ui.save TSDF", false, false);
  pangolin::Var<bool> fuseTSDF("ui.fuse TSDF",true,true);

  pangolin::Var<float> tsdfMu("ui.mu",0.5,0.,1.);
  pangolin::Var<int>   tsdfSliceD("ui.TSDF slice D",dTSDF/2,0,dTSDF-1);
  pangolin::Var<float> grid0x("ui.grid0 x",-3.0,-2.,0);
  pangolin::Var<float> grid0y("ui.grid0 y",-3.0,-2.,0);
  pangolin::Var<float> grid0z("ui.grid0 z",-3.0,-2.,0);
  pangolin::Var<float> gridEx("ui.gridE x", 3.0,2,3);
  pangolin::Var<float> gridEy("ui.gridE y", 3.0,2,3);
  pangolin::Var<float> gridEz("ui.gridE z", 3.0,2,3);

  pangolin::Var<bool> resetICP("ui.reset ICP",false,false);
  pangolin::Var<bool>  runICP("ui.run ICP", true, true);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);
  pangolin::Var<int>   icpErrorLvl("ui.ICP error vis lvl",0,0,2);

  pangolin::Var<bool> runSAM("ui.run SAM",true,true);
  pangolin::Var<bool> samIMU("ui.add IMU SAM",true,true);

  pangolin::Var<bool> showPcModel("ui.show model",true,true);
  pangolin::Var<bool> showPcCurrent("ui.show current",true,true);
  pangolin::Var<bool> showPcView("ui.show overview",true,true);

  Stopwatch::getInstance().setCustomSignature(12439347412);

  gui.verbose = true;

  tdp::SE3<float> T_mo(Eigen::Matrix4f::Identity());
  tdp::SE3f T_mo_0 = T_mo;
  tdp::SE3f T_mo_prev = T_mo_0;
  tdp::SE3f T_wr_imu_prev;
  size_t numFused = 0;

  tdp::ThreadedValue<bool> runWorker(true);
  std::thread workThread([&]() {
        while(runWorker.Get()) {
          if (pangolin::Pushed(saveTSDF)) {
            tdp::ManagedHostVolume<tdp::TSDFval> tmpTSDF(wTSDF, hTSDF, dTSDF);
            tmpTSDF.CopyFrom(cuTSDF, cudaMemcpyDeviceToHost);
            std::cout << "start writing TSDF to " << tsdfOutputPath << std::endl;
            tdp::SaveVolume(tmpTSDF, tsdfOutputPath);
            std::cout << "done writing TSDF to " << tsdfOutputPath << std::endl;
          }
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      });


  // Assemble initial quaternion through gtsam constructor ::quaternion(w,x,y,z);
  Rot3 prior_rotation = Rot3::Quaternion(1,0,0,0);
  Point3 prior_point(0,0,0);
  Pose3 prior_pose(prior_rotation, prior_point);
  Vector3 prior_velocity(0,0,0);
  imuBias::ConstantBias prior_imu_bias; // assume zero initial bias

  Values initial_values;
  int correction_count = 0;
  initial_values.insert(X(correction_count), prior_pose);
  initial_values.insert(V(correction_count), prior_velocity);
  initial_values.insert(B(correction_count), prior_imu_bias);  

  // Assemble prior noise model and add it the graph.
  noiseModel::Diagonal::shared_ptr pose_noise_model = 
    noiseModel::Diagonal::Sigmas((Vector(6) 
          << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished()); // rad,rad,rad,m, m, m
  noiseModel::Diagonal::shared_ptr velocity_noise_model = 
    noiseModel::Isotropic::Sigma(3,0.1); // m/s
  noiseModel::Diagonal::shared_ptr bias_noise_model = 
    noiseModel::Isotropic::Sigma(6,1e-3);

  // Add all prior factors (pose, velocity, bias) to the graph.
  NonlinearFactorGraph *graph = new NonlinearFactorGraph();
  graph->add(PriorFactor<Pose3>(X(correction_count), prior_pose,
        pose_noise_model));
  graph->add(PriorFactor<Vector3>(V(correction_count),
        prior_velocity,velocity_noise_model));
  graph->add(PriorFactor<imuBias::ConstantBias>(B(correction_count),
        prior_imu_bias,bias_noise_model));

  // We use the sensor specs to build the noise model for the IMU factor.
  double accel_noise_sigma = 0.0003924;
  double gyro_noise_sigma = 0.000205689024915;
  double accel_bias_rw_sigma = 0.004905;
  double gyro_bias_rw_sigma = 0.000001454441043;
  Matrix33 measured_acc_cov = Matrix33::Identity(3,3) * pow(accel_noise_sigma,2);
  Matrix33 measured_omega_cov = Matrix33::Identity(3,3) * pow(gyro_noise_sigma,2);
  Matrix33 integration_error_cov = Matrix33::Identity(3,3)*1e-8; // error committed in integrating position from velocities
  Matrix33 bias_acc_cov = Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2);
  Matrix33 bias_omega_cov = Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2);
  Matrix66 bias_acc_omega_int = Matrix::Identity(6,6)*1e-5; // error in the bias used for preintegration

  boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedD(0.0);
  // PreintegrationBase params:
  p->accelerometerCovariance = measured_acc_cov; // acc white noise in continuous
  p->integrationCovariance = integration_error_cov; // integration uncertainty continuous
  // should be using 2nd order integration
  // PreintegratedRotation params:
  p->gyroscopeCovariance = measured_omega_cov; // gyro white noise in continuous
  // PreintegrationCombinedMeasurements params:
  p->biasAccCovariance = bias_acc_cov; // acc bias in continuous
  p->biasOmegaCovariance = bias_omega_cov; // gyro bias in continuous
  p->biasAccOmegaInt = bias_acc_omega_int;
  
  PreintegrationType *imu_preintegrated_;
#ifdef USE_COMBINED
  imu_preintegrated_ = new PreintegratedCombinedMeasurements(p, prior_imu_bias);
#else
  imu_preintegrated_ = new PreintegratedImuMeasurements(p, prior_imu_bias);
#endif

  // Store previous state for the imu integration and the latest predicted outcome.
  NavState prev_state(prior_pose, prior_velocity);
  NavState prop_state = prev_state;
  imuBias::ConstantBias prev_bias = prior_imu_bias;

  // Keep track of the total error over the entire run for a simple performance metric.
  double current_position_error = 0.0, current_orientation_error = 0.0;

  double output_time = 0.0;
  double dt = 0.005;  // The real system has noise, but here, results are nearly 
                      // exactly the same, so keeping this for simplicity.
                      //
  //
  tdp::ThreadedValue<bool> receiveImu(true);
  tdp::ThreadedValue<size_t> numReceived(0);
  tdp::ThreadedValue<int64_t> t_host_video(0);
  std::thread receiverThread (
    [&]() {
      tdp::ImuObs imuObs;
      tdp::ImuObs imuObsPrev;
      bool calibrated = false;
      Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
      while(receiveImu.Get()) {
        if (imu->GrabNext(imuObs)) {
          while(t_host_video.Get() < imuObs.t_host) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
          }
          double dt = (t_host_video.Get() - imuObs.t_host)*1e-9;
          if (numReceived.Get() > 0 && dt < 0.1)  {
            double dt = (imuObs.t_device- imuObsPrev.t_device)*1e-9;
            std::cout << "imu observation " <<  imuObs.t_host
              << " dt: " << dt << std::endl;
            imu_preintegrated_->integrateMeasurement(
              (T_ir.rotation().Inverse()*imuObs.acc).cast<double>(),
              (T_ir.rotation().Inverse()*imuObs.omega).cast<double>(), 
              dt);
          }
          imuObsPrev = imuObs;
          numReceived.Increment();
        }
        //std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });

  tdp::SE3f T_mo_isam; 

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    if (runFusion.GuiChanged() && !runFusion) {
      T_mo_0 = tdp::SE3f();
      T_mo = T_mo_0;
      T_mo_prev = T_mo_0;
    }

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();
    tdp::Image<uint16_t> dRaw;
    int64_t t_host_us_d = 0;
    if (!gui.ImageD(dRaw,0,&t_host_us_d)) continue;
    t_host_video.Set(t_host_us_d*1000);
//    tdp::SE3f T_wr_imu = T_ir.Inverse() * imuInterp.Ts_wi_[t_host_us_d*1000]*T_ir;
    std::cout << " depth frame at " << t_host_us_d << " us" << std::endl;

    tdp::Vector3fda grid0(grid0x,grid0y,grid0z);
    tdp::Vector3fda gridE(gridEx,gridEy,gridEz);
    tdp::Vector3fda dGrid = gridE - grid0;
    dGrid(0) /= (wTSDF-1);
    dGrid(1) /= (hTSDF-1);
    dGrid(2) /= (dTSDF-1);

    if (gui.verbose) std::cout << "setup pyramids" << std::endl;
    TICK("Setup Pyramids");
    cuDraw.CopyFrom(dRaw, cudaMemcpyHostToDevice);
    ConvertDepthGpu(cuDraw, cuD, depthSensorScale, tsdfDmin, tsdfDmax);
    // construct pyramid  
    tdp::ConstructPyramidFromImage<float,3>(cuD, cuDPyr,
        cudaMemcpyDeviceToDevice, 0.03);
    tdp::Depth2PCsGpu(cuDPyr,camD,pcs_c);
    // compute normals
    tdp::Depth2Normals(cuDPyr,camD,ns_c);
    TOCK("Setup Pyramids");

//    if (icpImu && imu) 
//      T_mo = (T_wr_imu * T_wr_imu_prev.Inverse()) * T_mo;

    if (runICP && (!runFusion || numFused > 30)) {
      if (gui.verbose) std::cout << "icp" << std::endl;
      TICK("ICP");
      std::vector<size_t> maxIt{icpIter0,icpIter1,icpIter2};
      tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, T_mo, tdp::SE3f(),
          camD, maxIt, icpAngleThr_deg, icpDistThr); 
      TOCK("ICP");

//      std::cout << "T_mo update: " << std::endl 
//        << T_mo * T_mo_prev.Inverse() << std::endl;
//      std::cout << "IMU : " << std::endl 
//        << T_wr_imu * T_wr_imu_prev.Inverse() << std::endl;
    }
    std::cout << "T_mo after ICP: " << std::endl 
      << T_mo  << std::endl;



    if (runSAM && numFused > 30) {

      correction_count++;
      if (samIMU) {
        // Adding IMU factor and GPS factor and optimizing.
#ifdef USE_COMBINED
        PreintegratedCombinedMeasurements *preint_imu_combined = dynamic_cast<PreintegratedCombinedMeasurements*>(imu_preintegrated_);
        CombinedImuFactor imu_factor(X(correction_count-1), V(correction_count-1),
            X(correction_count  ), V(correction_count  ),
            B(correction_count-1), B(correction_count  ),
            *preint_imu_combined);
        graph->add(imu_factor);
#else
        PreintegratedImuMeasurements *preint_imu = dynamic_cast<PreintegratedImuMeasurements*>(imu_preintegrated_);
        ImuFactor imu_factor(X(correction_count-1), V(correction_count-1),
            X(correction_count  ), V(correction_count  ),
            B(correction_count-1),
            *preint_imu);
        graph->add(imu_factor);
        imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
        graph->add(BetweenFactor<imuBias::ConstantBias>(B(correction_count-1), 
              B(correction_count  ), 
              zero_bias, bias_noise_model));

        preint_imu->print();
#endif
      }

      noiseModel::Diagonal::shared_ptr correction_noise = noiseModel::Isotropic::Sigma(6,0.01);

      tdp::SE3f dT_icp = T_mo * T_mo_prev.Inverse();

      std::cout << "adding ICP factor " 
        << std::endl 
        << dT_icp.matrix3x4() << std::endl;
      BetweenFactor<Pose3> icp_factor(X(correction_count-1),
          X(correction_count),
          Pose3(dT_icp.matrix().cast<double>()),
          correction_noise);
//      GPSFactor gps_factor(X(correction_count),
//                           Point3(gps(0),  // N,
//                                  gps(1),  // E,
//                                  gps(2)), // D,
//                           correction_noise);
//      graph->add(gps_factor);
      graph->add(icp_factor);      

      if (samIMU) {
        // Now optimize and compare results.
        prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
        initial_values.insert(X(correction_count), prop_state.pose());
        initial_values.insert(V(correction_count), prop_state.v());
        initial_values.insert(B(correction_count), prev_bias);
      } else {
        initial_values.insert(X(correction_count), Pose3(T_mo.matrix().cast<double>()));
        initial_values.insert(V(correction_count), Vector3(0.,0.,0.));
        initial_values.insert(B(correction_count), prev_bias);
      }

      std::cout << "now optimizing" << std::endl;
      LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
      Values result = optimizer.optimize();

      if (samIMU) {
        // Overwrite the beginning of the preintegration for the next step.
        prev_state = NavState(result.at<Pose3>(X(correction_count)),
            result.at<Vector3>(V(correction_count)));
        prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));
        // Reset the preintegration object.
        imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);
      }

      // Print out the position and orientation error for comparison.
      T_mo_isam = tdp::SE3f(prev_state.pose().matrix().cast<float>());

      std::cout << "ISAM " 
        << std::endl 
        << T_mo_isam.matrix3x4() << std::endl;

      std::ofstream out("./graph.viz");
      graph->saveGraph(out);
      out.close();
    }

    if (runFusion && (fuseTSDF || numFused <= 30)) {
      if (gui.verbose) std::cout << "add to tsdf" << std::endl;
      TICK("Add To TSDF");
      AddToTSDF(cuTSDF, cuD, T_mo, camD, grid0, dGrid, tsdfMu); 
      numFused ++;
      TOCK("Add To TSDF");
    }

    if (runFusion) {
      if (gui.verbose) std::cout << "ray trace" << std::endl;
      TICK("Ray Trace TSDF");
      RayTraceTSDF(cuTSDF, pcs_m.GetImage(0), 
          ns_m.GetImage(0), T_mo, camD, grid0, dGrid, tsdfMu); 
      // get pc in model coordinate system
      tdp::CompletePyramid<tdp::Vector3fda,3>(pcs_m, cudaMemcpyDeviceToDevice);
      TOCK("Ray Trace TSDF");
      tdp::CompleteNormalPyramid<3>(ns_m, cudaMemcpyDeviceToDevice);
    }

    if (pangolin::Pushed(resetTSDF)) {
      TSDF.Fill(tdp::TSDFval(-1.01,0.));
      dEst.Fill(0.);
      cuTSDF.CopyFrom(TSDF, cudaMemcpyHostToDevice);
      numFused = 0;
      T_mo = T_mo_0;
      T_mo_prev = T_mo;
      std::cout << "resetting ICP and TSDF" << std::endl;
    }
    if (pangolin::Pushed(resetICP)) {
      T_mo = T_mo_0;
      T_mo_prev = T_mo;
      std::cout << "resetting ICP" << std::endl;
    }

    if (gui.verbose) std::cout << "draw 3D" << std::endl;

    TICK("Draw 3D");
    // Render point cloud from viewpoint of origin
    tdp::SE3f T_mv;
    RayTraceTSDF(cuTSDF, cuDView, nEstdummy, T_mv, camView, grid0,
        dGrid, tsdfMu); 
    tdp::Depth2PCGpu(cuDView,camView,cuPcView);

    glEnable(GL_DEPTH_TEST);
    viewPc3D.Activate(s_cam);

    Eigen::AlignedBox3f box(grid0,gridE);
    glColor4f(1,0,0,0.5f);
    pangolin::glDrawAlignedBox(box);

    // imu
//    pangolin::glSetFrameOfReference(T_wr_imu.matrix());
//    pangolin::glDrawAxis(0.2f);
//    pangolin::glUnsetFrameOfReference();

    // render global view of the model first
    pangolin::glDrawAxis(0.1f);
    if (showPcView) {
      {
        pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
        cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
        cudaMemcpy(*cuPcbufp, cuPcView.ptr_, cuPcView.SizeBytes(),
            cudaMemcpyDeviceToDevice);
      }
      glColor3f(0,0,1);
      pangolin::RenderVbo(cuPcbuf);
    }
    // render model and observed point cloud
    if (showPcModel) {
      {
        pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
        cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
        tdp::Image<tdp::Vector3fda> pc0 = pcs_m.GetImage(icpErrorLvl);
        cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
            cudaMemcpyDeviceToDevice);
      }
      glColor3f(0,1,0);
      pangolin::RenderVbo(cuPcbuf);
    }

    pangolin::glDrawAxis(T_mo_isam.matrix(), 0.15f);

    pangolin::glSetFrameOfReference(T_mo.matrix());
    // render current camera second in the propper frame of
    // reference
    pangolin::glDrawAxis(0.1f);
    if (showPcCurrent) {
      {
        pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
        cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
        tdp::Image<tdp::Vector3fda> pc0 = pcs_c.GetImage(icpErrorLvl);
        cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
            cudaMemcpyDeviceToDevice);
      }
      glColor3f(1,0,0);
      pangolin::RenderVbo(cuPcbuf);
    }
    pangolin::glUnsetFrameOfReference();


    viewN3D.Activate(s_cam);
    // render global view of the model first
    pangolin::glDrawAxis(0.1f);
    if (showPcView) {
      {
        pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
        cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
        cudaMemcpy(*cuPcbufp, nEstdummy.ptr_, nEstdummy.SizeBytes(),
            cudaMemcpyDeviceToDevice);
      }
      glColor3f(0,0,1);
      pangolin::RenderVbo(cuPcbuf);
    }
    // render model and observed point cloud
    if (showPcModel) {
      {
        pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
        cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
        tdp::Image<tdp::Vector3fda> pc0 = ns_m.GetImage(icpErrorLvl);
        cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
            cudaMemcpyDeviceToDevice);
      }
      glColor3f(0,1,0);
      pangolin::RenderVbo(cuPcbuf);
    }
    Eigen::Matrix4f R_mo = T_mo.matrix();
    R_mo.topRightCorner(3,1).fill(0.);
    pangolin::glSetFrameOfReference(R_mo);
    // render current camera second in the propper frame of
    // reference
    pangolin::glDrawAxis(0.1f);
    if (showPcCurrent) {
      {
        pangolin::CudaScopedMappedPtr cuPcbufp(cuPcbuf);
        cudaMemset(*cuPcbufp,0,hc*wc*sizeof(tdp::Vector3fda));
        tdp::Image<tdp::Vector3fda> pc0 = ns_c.GetImage(icpErrorLvl);
        cudaMemcpy(*cuPcbufp, pc0.ptr_, pc0.SizeBytes(),
            cudaMemcpyDeviceToDevice);
      }
      glColor3f(1,0,0);
      pangolin::RenderVbo(cuPcbuf);
    }
    pangolin::glUnsetFrameOfReference();
    TOCK("Draw 3D");

    TICK("Draw 2D");
    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);
    gui.ShowFrames();

    tsdfDEst.CopyFrom(cuDEst,cudaMemcpyDeviceToHost);
    viewTsdfDEst.SetImage(tsdfDEst);

    tdp::Image<tdp::TSDFval> cuTsdfSlice =
      cuTSDF.GetImage(std::min((int)cuTSDF.d_-1,tsdfSliceD.Get()));
    tdp::ManagedHostImage<tdp::TSDFval> tsdfSliceRaw(cuTsdfSlice.w_, 
        cuTsdfSlice.h_);
    tsdfSliceRaw.CopyFrom(cuTsdfSlice,cudaMemcpyDeviceToHost);
    for (size_t i=0; i<tsdfSliceRaw.Area(); ++i) 
      tsdfSlice[i] = tsdfSliceRaw[i].f;
    viewTsdfSliveView.SetImage(tsdfSlice);

    tdp::PyramidToImage<float,3>(cuDPyr,dispDepthPyr,
        cudaMemcpyDeviceToHost);
    viewDepthPyr.SetImage(dispDepthPyr);

    if (dispNormalsPyrEst) {
      tdp::PyramidToImage<tdp::Vector3fda,3>(ns_m,cuDispNormalsPyr,
          cudaMemcpyDeviceToDevice);
    } else {
      tdp::PyramidToImage<tdp::Vector3fda,3>(ns_c,cuDispNormalsPyr,
          cudaMemcpyDeviceToDevice);
    }
    tdp::Normals2Image(cuDispNormalsPyr, cuDispNormals2dPyr);
    dispNormals2dPyr.CopyFrom(cuDispNormals2dPyr,cudaMemcpyDeviceToHost);
    viewNormalsPyr.SetImage(dispNormals2dPyr);

    TOCK("Draw 2D");

    if (!runFusion) {
      tdp::SO3f R_mo = T_mo.rotation();
      for (size_t lvl=0; lvl<3; ++lvl) {
        tdp::Image<tdp::Vector3fda> pc = pcs_c.GetImage(lvl);
        tdp::Image<tdp::Vector3fda> n = ns_c.GetImage(lvl);
        tdp::TransformPc(T_mo, pc);
        tdp::TransformPc(R_mo, n);
      }
      pcs_m.CopyFrom(pcs_c,cudaMemcpyDeviceToDevice);
      ns_m.CopyFrom(ns_c,cudaMemcpyDeviceToDevice);
    }
    if (!gui.paused()) {
//      T_wr_imu_prev = T_wr_imu;
      T_mo_prev = T_mo;
    }

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    Stopwatch::getInstance().sendAll();
    pangolin::FinishFrame();
  }
  runWorker.Set(false);
  workThread.join();
  receiveImu.Set(false);
  receiverThread.join();
  if (imu) imu->Stop();
  delete imu;
  return 0;
}

