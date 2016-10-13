/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image_io.h>

#include <tdp/eigen/dense.h>
#include <tdp/data/managed_image.h>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>
#include <tdp/preproc/normals.h>

#include <tdp/tsdf/tsdf.h>
#include <tdp/data/managed_volume.h>
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>
#include <tdp/preproc/curvature.h>

#include <vector>
#include <list>
#include <tdp/bb/bb.h>

#include <tdp/distributions/normal_mm.h>
#include <tdp/distributions/vmf_mm.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/render.h>

#include <tdp/icp/icp.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/managed_pyramid.h>

typedef tdp::CameraPoly3<float> CameraT;
//typedef tdp::Camera<float> CameraT;

int main( int argc, char* argv[] )
{
  const std::string inputA = std::string(argv[1]);
  const std::string inputB = std::string(argv[2]);
  const std::string option = (argc > 3) ? std::string(argv[3]) : "";

  bool runOnce = false;
  if (!option.compare("-1")) {
    runOnce = true; 
  }

  // Create OpenGL window - guess sensible dimensions
  int menue_w = 180;
  pangolin::CreateWindowAndBind( "GuiBase", 1200+menue_w, 800);
  // current frame in memory buffer and displaying.
  pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menue_w));
  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // setup container
  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
    .SetBounds(0., 1.0, pangolin::Attach::Pix(menue_w), 1.0);
  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& viewPc = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewPc);
  pangolin::View& viewNA = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewNA);
  pangolin::View& viewNB = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(viewNB);
  // use those OpenGL buffers
  
  tdp::ManagedHostImage<tdp::Vector3fda> pcA, pcB;
  tdp::ManagedHostImage<tdp::Vector3fda> nsA, nsB;
  tdp::LoadPointCloud(inputA, pcA, nsA);
  tdp::LoadPointCloud(inputB, pcB, nsB);

  tdp::ManagedHostImage<uint16_t> zA(pcA.w_,pcA.h_), zB(pcB.w_,pcB.h_);
  tdp::ManagedDeviceImage<uint16_t> cuZA(pcA.w_,pcA.h_), cuZB(pcB.w_,pcB.h_);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuNsA(pcA.w_,pcA.h_), cuNsB(pcB.w_,pcB.h_);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuPcA(pcA.w_,pcA.h_), cuPcB(pcB.w_,pcB.h_);

  cuPcA.CopyFrom(pcA, cudaMemcpyHostToDevice);
  cuPcB.CopyFrom(pcB, cudaMemcpyHostToDevice);
  cuNsA.CopyFrom(nsA, cudaMemcpyHostToDevice);
  cuNsB.CopyFrom(nsB, cudaMemcpyHostToDevice);

  tdp::ManagedHostImage<int> assoc_ba(pcA.w_,pcA.h_);
  tdp::ManagedDeviceImage<int> cuAssoc_ba(pcA.w_,pcA.h_);

  pangolin::GlBuffer vboA, vboB;
  vboA.Reinitialise(pangolin::GlArrayBuffer, pcA.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  vboA.Upload(pcA.ptr_, pcA.SizeBytes(), 0);
  vboB.Reinitialise(pangolin::GlArrayBuffer, pcB.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  vboB.Upload(pcB.ptr_, pcB.SizeBytes(), 0);
  pangolin::GlBuffer nboA, nboB;
  nboA.Reinitialise(pangolin::GlArrayBuffer, nsA.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  nboA.Upload(nsA.ptr_, nsA.SizeBytes(), 0);
  nboB.Reinitialise(pangolin::GlArrayBuffer, nsB.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  nboB.Upload(nsB.ptr_, nsB.SizeBytes(), 0);

  pangolin::GlBuffer valueboA, valueboB;
  valueboA.Reinitialise(pangolin::GlArrayBuffer, pcA.Area(),
      GL_UNSIGNED_SHORT, 1, GL_DYNAMIC_DRAW);
  valueboB.Reinitialise(pangolin::GlArrayBuffer, pcB.Area(),
      GL_UNSIGNED_SHORT, 1, GL_DYNAMIC_DRAW);

  pangolin::Var<bool> computevMFMMs("ui.compute vMFMMs", false, false);
  pangolin::Var<bool> computeGMMs("ui.compute GMMs", false, false);

  pangolin::Var<int> maxIt("ui.max Iter", 100, 1, 100);
  pangolin::Var<float> minNchangePerc("ui. min change perc", 0.03, 0.001, 0.1);
  pangolin::Var<float> lambdaS2("ui.lambda S2", 55., 10., 180.);
  pangolin::Var<float> lambdaR3("ui.lambda R3", 1.0, 0.5, 2.0);

  pangolin::Var<bool> computeAlignment("ui.align", false, false);
  pangolin::Var<int>   maxLvlRot("ui.max lvl rot",12,1,15);
  pangolin::Var<int>   maxLvlTrans("ui.max lvl trans",20,10,22);
  pangolin::Var<int>   maxItBB("ui.max Iter BB", 3000, 1000, 5000);

  pangolin::Var<bool> computeProjectiveICP("ui.projtive ICP", false, false);
  pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",15,0.,90.);
  pangolin::Var<float> icpDistThr("ui.icp dist thr",0.10,0.,1.);
  pangolin::Var<int>   icpIter0("ui.ICP iter lvl 0",10,0,10);
  pangolin::Var<int>   icpIter1("ui.ICP iter lvl 1",7,0,10);
  pangolin::Var<int>   icpIter2("ui.ICP iter lvl 2",5,0,10);

  std::vector<tdp::Normal3d> gmmA, gmmB;
  std::vector<tdp::vMF3d> vmfmmA, vmfmmB;
  tdp::DPmeans dpmeansA(lambdaR3), dpmeansB(lambdaR3);
  tdp::DPvMFmeans dpvmfmeansA(cos(lambdaS2*M_PI/180.)), 
                  dpvmfmeansB(cos(lambdaS2*M_PI/180.));

  Eigen::Vector3f minAB, maxAB;
  BoundingBox(pcA, minAB, maxAB, true);
  BoundingBox(pcB, minAB, maxAB, false);

  tdp::SE3f T_ab;

  float maxVal, minVal;
  minVal = -1.f;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (runOnce) break;

    if (pangolin::Pushed(computevMFMMs)) {
      std::cout << " lambda " << lambdaS2 << std::endl;
      dpvmfmeansA.lambda_ = cos(lambdaS2*M_PI/180.);
      dpvmfmeansB.lambda_ = cos(lambdaS2*M_PI/180.);

      std::cout << "computing vMF MM A" << std::endl;
      tdp::ComputevMFMM<double>(nsA, cuNsA, dpvmfmeansA, maxIt, minNchangePerc,
          zA, cuZA, vmfmmA);
      for (auto& vmf : vmfmmA) vmf.Print();

      std::cout << "computing vMF MM B" << std::endl;
      tdp::ComputevMFMM<double>(nsB, cuNsB, dpvmfmeansB, maxIt, minNchangePerc,
          zB, cuZB, vmfmmB);
      for (auto& vmf : vmfmmB) vmf.Print();

      maxVal = vmfmmA.size();
      valueboA.Upload(zA.ptr_, zA.SizeBytes(), 0);
      valueboB.Upload(zB.ptr_, zB.SizeBytes(), 0);
    }

    if (pangolin::Pushed(computeGMMs)) {
      dpmeansA.lambda_ = lambdaR3;
      dpmeansB.lambda_ = lambdaR3;

      std::cout << "computing GMM A" << std::endl;
      tdp::ComputeGMMfromPC<double>(pcA, cuPcA, 
          dpmeansA, maxIt, minNchangePerc, zA, cuZA, gmmA);
      for (auto& g : gmmA) g.Print();

      std::cout << "computing GMM B" << std::endl;
      tdp::ComputeGMMfromPC<double>(pcB, cuPcB, 
          dpmeansB, maxIt, minNchangePerc, zB, cuZB, gmmB);
      for (auto& g : gmmB) g.Print();

      maxVal = gmmA.size();
      valueboA.Upload(zA.ptr_, zA.SizeBytes(), 0);
      valueboB.Upload(zB.ptr_, zB.SizeBytes(), 0);
    }
    
    if (pangolin::Pushed(computeAlignment)) {
			tdp::LowerBoundS3d lower_bound_S3(vmfmmA, vmfmmB);
			tdp::UpperBoundIndepS3d upper_bound_S3(vmfmmA, vmfmmB);
			tdp::UpperBoundConvexS3d upper_bound_convex_S3(vmfmmA, vmfmmB);

			std::list<tdp::NodeS3d> nodesS3;
			Eigen::Quaterniond q_star;
			double lb_star = 1e99;
			double eps = 1e-8;

      std::cout << "Tesselating Sphere for initial nodes" << std::endl;
			nodesS3 = tdp::GenerateNotesThatTessellateS3<double>();

			tdp::BranchAndBound<double,tdp::NodeS3d> bb(lower_bound_S3, 
          upper_bound_convex_S3);
      std::cout << "Running B&B for Rotation " 
        << " #nodes0 " << nodesS3.size() << std::endl;
			tdp::NodeS3d node_star = bb.Compute(nodesS3, eps, maxLvlRot, maxItBB);
			q_star = node_star.GetLbArgument();
			lb_star = node_star.GetLB();

      std::cout << " optimal rotation: "
        << std::endl << q_star.matrix() << std::endl;

			std::list<tdp::NodeR3d> nodesR3 =
				tdp::GenerateNotesThatTessellateR3<double>(
            minAB.cast<double>(), maxAB.cast<double>(), 
            static_cast<double>((maxAB-minAB).norm()));
			tdp::LowerBoundR3d lower_bound_R3(gmmA, gmmB, q_star);
			tdp::UpperBoundIndepR3d upper_bound_R3(gmmA, gmmB, q_star);
			tdp::UpperBoundConvexR3d upper_bound_convex_R3(gmmA, gmmB, q_star);

			eps = 1e-9;
			tdp::BranchAndBound<double,tdp::NodeR3d> bbR3(lower_bound_R3, 
          upper_bound_convex_R3);
			tdp::NodeR3d nodeR3_star = bbR3.Compute(nodesR3, eps, maxLvlTrans, maxItBB);
			Eigen::Vector3d t =  nodeR3_star.GetLbArgument();

			T_ab = tdp::SE3f(q_star.matrix().cast<float>(), t.cast<float>());
    }

    if (pangolin::Pushed(computeProjectiveICP)) {
        size_t maxIt = icpIter0;
      for (size_t it=0; it<maxIt; ++it) {
        //    tdp::TransformPc(T_ab, cuPcB);
        //    tdp::TransformPc(T_ab.rotation(), cuNsB);
        tdp::AssociateANN(pcA, pcB, T_ab.Inverse(), assoc_ba);
        cuAssoc_ba.CopyFrom(assoc_ba, cudaMemcpyHostToDevice);

        float err;
        float count;

        tdp::ICP::ComputeGivenAssociation(cuPcA, cuNsA, cuPcB, cuNsB,
            cuAssoc_ba, T_ab, 1, icpAngleThr_deg, icpDistThr,
            err, count);

        std::cout << T_ab.matrix3x4() << std::endl;
      }
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    if (viewPc.IsShown()) {
      viewPc.Activate(s_cam);
      pangolin::glDrawAxis(0.1);

      glPointSize(1.);
      tdp::RenderLabeledVbo(vboA, valueboA, s_cam, maxVal);

      pangolin::glSetFrameOfReference(T_ab.matrix());
      pangolin::glDrawAxis(0.1);
      tdp::RenderLabeledVbo(vboB, valueboB, s_cam, maxVal);
      pangolin::glUnsetFrameOfReference();
    }
    if (viewNA.IsShown()) {
      viewNA.Activate(s_cam);
      // draw the axis
      pangolin::glDrawAxis(0.1);
      glPointSize(1.);
      tdp::RenderLabeledVbo(nboA, valueboA, s_cam, maxVal);
    }

    if (viewNB.IsShown()) {
      viewNB.Activate(s_cam);
      tdp::SE3f T(T_ab);
      T.matrix().topRightCorner(3,1).fill(0.);
      pangolin::glSetFrameOfReference(T.matrix());
      pangolin::glDrawAxis(0.1);
      tdp::RenderLabeledVbo(nboB, valueboB, s_cam, maxVal);
      pangolin::glUnsetFrameOfReference();
    }

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}

