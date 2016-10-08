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
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/tsdf/tsdf.h>
#include <tdp/data/managed_volume.h>
#include <pangolin/utils/timer.h>

#include <tdp/io/tinyply.h>
#include <tdp/preproc/curvature.h>
#include <tdp/distributions/normal_mm.h>
#include <tdp/distributions/vmf_mm.h>

#include <tdp/bb/bb.h>

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
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  container.AddDisplay(d_cam);
  // use those OpenGL buffers
  
  tdp::ManagedHostImage<tdp::Vector3fda> vertsA, vertsB;
  tdp::ManagedHostImage<tdp::Vector3fda> nsA, nsB;
  tdp::LoadPointCloud(inputA, vertsA, nsA);
  tdp::LoadPointCloud(inputB, vertsB, nsB);

  tdp::ManagedHostImage<uint16_t> zA(vertsA.w_,vertsA.h_), zB(vertsB.w_,vertsB.h_);
  tdp::ManagedDeviceImage<uint16_t> cuZA(vertsA.w_,vertsA.h_), cuZB(vertsB.w_,vertsB.h_);

  pangolin::GlBuffer vboA, vboB;
  vboA.Reinitialise(pangolin::GlArrayBuffer, vertsA.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  vboA.Upload(vertsA.ptr_, vertsA.SizeBytes(), 0);
  vboB.Reinitialise(pangolin::GlArrayBuffer, vertsB.Area(),  GL_FLOAT,
      3, GL_DYNAMIC_DRAW);
  vboB.Upload(vertsB.ptr_, vertsB.SizeBytes(), 0);
//  pangolin::GlBuffer valuebo;
//  valuebo.Reinitialise(pangolin::GlArrayBuffer, vertices.w_,  GL_FLOAT,
//      1, GL_DYNAMIC_DRAW);

  // load and compile shader
  std::string shaderRoot = SHADER_DIR;
  pangolin::GlSlProgram progValueShading;
  progValueShading.AddShaderFromFile(pangolin::GlSlVertexShader, 
      shaderRoot+std::string("valueShading.vert"));
  progValueShading.AddShaderFromFile(pangolin::GlSlFragmentShader,
      shaderRoot+std::string("valueShading.frag"));
  progValueShading.Link();

  pangolin::Var<bool> computeAlignment("ui.align", false, true);

  pangolin::Var<int> maxIt("ui.max Iter", 10, 1, 30);
  pangolin::Var<float> minNchangePerc("ui. min change perc", 0.03, 0.001, 0.1);

  pangolin::Var<float> lambdaS2("ui.lambda S2", 30.*M_PI/180., 10.*M_PI/180., 180.*M_PI/180.);
  pangolin::Var<float> lambdaR3("ui.lambda R3", 0.3, 0.1, 1.0);


  std::vector<tdp::Normal3f> gmmA, gmmB;
  std::vector<tdp::vMF3f> vmfmmA, vmfmmB;
  tdp::DPmeans dpmeansA(lambdaR3), dpmeansB(lambdaR3);
  tdp::DPvMFmeans dpvmfmeansA(lambdaS2), dpvmfmeansB(lambdaS2);

  tdp::SE3f T_ab;

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (runOnce) break;
    
    if (pangolin::Pushed(computeAlignment)) {
      tdp::ComputevMFMM(nsA, cuNsA, dpvmfmeansA, maxIt, minNchangePerc,
          zA, cuZA, vmfmmA);
      tdp::ComputevMFMM(nsB, cuNsB, dpvmfmeansB, maxIt, minNchangePerc,
          zB, cuZB, vmfmmB);

			tdp::LowerBoundS3 lower_bound_S3(vmfmmA, vmfmmB);
			tdp::UpperBoundIndepS3 upper_bound_S3(vmfmmA, vmfmmB);
			tdp::UpperBoundConvexS3 upper_bound_convex_S3(vmfmmA, vmfmmB);

			std::list<tdp::NodeS3> nodesS3;
			Eigen::Quaterniond q_star;
			double lb_star = 1e99;
			double eps = 1e-8;
			//  double eps = 8e-7;
			uint32_t max_lvl = 12;
			uint32_t max_it = 5000;

			nodesS3 = tdp::GenerateNotesThatTessellateS3();
			tdp::BranchAndBound<tdp::NodeS3> bb(lower_bound_S3, upper_bound_convex_S3);
			tdp::NodeS3 node_star = bb.Compute(nodesS3, eps, max_lvl, max_it);
			q_star = node_star.GetLbArgument();
			lb_star = node_star.GetLB();

      tdp::ComputeGMMfromPC(vertsA, cuVertsA, 
          dpmeansA, maxIt, minNchangePerc, zA, cuZA, gmmA);
      tdp::ComputeGMMfromPC(vertsB, cuVertsB, 
          dpmeansB, maxIt, minNchangePerc, zB, cuZB, gmmB);

			std::list<tdp::NodeR3> nodesR3 =
				tdp::GenerateNotesThatTessellateR3(min, max, (max-min).norm());
			tdp::LowerBoundR3 lower_bound_R3(gmmA, gmmB, q);
			tdp::UpperBoundIndepR3 upper_bound_R3(gmmA, gmmB, q);
			tdp::UpperBoundConvexR3 upper_bound_convex_R3(gmmA, gmmB, q);

			eps = 1e-9;
			max_it = 5000;
			max_lvl = 22;
			tdp::BranchAndBound<tdp::NodeR3> bbR3(lower_bound_R3, upper_bound_convex_R3);
			tdp::NodeR3 nodeR3_star = bbR3.Compute(nodesR3, eps, max_lvl, max_it);
			Eigen::Vector3d t =  nodeR3_star.GetLbArgument();

			T_ab = tdp::SE3f(q_star.matrix().cast<float>(), t.cast<float>());
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);

    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
    // draw the axis

    glColor4f(1.,0.,0.,1.);
    pangolin::glDrawAxis(0.1);
    pangolin::RenderVbo(vboA);

    pangolin::glSetFrameOfReference(T_ab.matrix());
    glColor4f(0.,1.,0.,1.);
    pangolin::glDrawAxis(0.1);
    pangolin::RenderVbo(vboB);
    pangolin::glUnsetFrameOfReference();

    glDisable(GL_DEPTH_TEST);
    // Draw 2D stuff

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
  }
  return 0;
}

