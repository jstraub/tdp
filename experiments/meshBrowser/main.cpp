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
#include <tdp/gl/shaders.h>
#include <tdp/gl/matcap.h>

int main( int argc, char* argv[] )
{
  const std::string input_uri = std::string(argv[1]);
  const std::string option = (argc > 2) ? std::string(argv[2]) : "";

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
//  pangolin::View& d_cam = pangolin::CreateDisplay()
//    .SetHandler(new pangolin::Handler3D(s_cam));
//  container.AddDisplay(d_cam);
  pangolin::Uri uri = pangolin::ParseUri(input_uri);
  std::vector<std::string> files;
  if (uri.scheme.compare("files") == 0) {
//    std::cout << uri.scheme << std::endl;
//    std::cout << uri.url << std::endl;
    pangolin::FilesMatchingWildcard(uri.url, files);
//    for (auto& file : files) 
//      std::cout << file << std::endl;
  } else {
    std::cout << "only supporting files:// uri so far" << std::endl;
    return 1;
  }
  
  size_t W = 3;
  size_t H = 3;

  if (files.size() < 3) {
    H = 1;
    W = files.size();
  } else if (files.size() < 5) {
    H = 2;
    W = 2;
  } else if (files.size() < 7) {
    H = 2;
    W = 3;
  }

  std::vector<std::string> viewNames(W*H,"");
  for (size_t i=0; i<W*H; ++i) {
    std::stringstream ss;
    ss << "view" << i;
    viewNames[i] = ss.str();
    pangolin::View& view = pangolin::Display(viewNames[i]);
    view.SetHandler(new pangolin::Handler3D(s_cam));
    container.AddDisplay(view);
  }

//  LoadPointCloud( const std::string& path,
//        ManagedHostImage<Vector3fda>& verts);


  tdp::ManagedHostImage<tdp::Vector3fda> verts;
  tdp::ManagedHostImage<tdp::Vector3fda> ns;
  int frame=0;
  int sliderPrev = 0;
  pangolin::Var<int> slider("ui.slide", 0, 0, files.size()-W*H);
  pangolin::Var<int> matcapId("ui.matcap", 4, 0, 10);

  std::vector<pangolin::GlBuffer*> vbos(W*H, nullptr);
  std::vector<pangolin::GlBuffer*> nbos(W*H, nullptr);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    // clear the OpenGL render buffers
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    if (frame == 0 || slider.GuiChanged()) {
      //TODO be smarter about loading
      for (size_t i=slider; i<slider+W*H; ++i) {
        tdp::LoadPointCloud(files[i], verts, ns);
        if (vbos[i]) delete vbos[i];
        vbos[i] = new pangolin::GlBuffer(pangolin::GlArrayBuffer,
            verts.w_,  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        vbos[i]->Upload(verts.ptr_, verts.SizeBytes(), 0);
        if (nbos[i]) delete nbos[i];
        nbos[i] = new pangolin::GlBuffer(pangolin::GlArrayBuffer,
            ns.w_,  GL_FLOAT, 3, GL_DYNAMIC_DRAW);
        nbos[i]->Upload(ns.ptr_, ns.SizeBytes(), 0);
      }
      sliderPrev = slider;
    }

    // Draw 3D stuff
    glEnable(GL_DEPTH_TEST);
    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
    for (size_t i=0; i<W*H; ++i) {
      pangolin::Display(viewNames[i]).Activate(s_cam);
      pangolin::glDrawAxis(0.1);
      if (vbos[i] && nbos[i]) {
        auto& shader = tdp::Shaders::Instance()->matcapShader_;
        shader.Bind();
        shader.SetUniform("P",P);
        shader.SetUniform("MV",MV);
        glEnable(GL_TEXTURE_2D);
        tdp::Matcap::Instance()->Bind(matcapId);
        shader.SetUniform("matcap",0);
        nbos[i]->Bind();
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0); 
        vbos[i]->Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glPointSize(1.);
        glDrawArrays(GL_POINTS, 0, vbos[i]->num_elements);
        tdp::Matcap::Instance()->Unbind();
        glDisable(GL_TEXTURE_2D);
        tdp::Shaders::Instance()->matcapShader_.Unbind();
        glDisableVertexAttribArray(1);
        nbos[i]->Unbind();
        glDisableVertexAttribArray(0);
        vbos[i]->Unbind();
        shader.Unbind();
      }
    }

    glDisable(GL_DEPTH_TEST);
    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    // finish this frame
    pangolin::FinishFrame();
    frame++;
  }
  return 0;
}

