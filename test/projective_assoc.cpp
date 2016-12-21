#include <tdp/testing/testing.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/data/managed_image.h>
#include <tdp/camera/projective_labels.h>
#include <bitset>

#include <pangolin/display/display.h>
#include <tdp/gui/quickView.h>

TEST(projectiveAssoc, init) {
  tdp::Cameraf::Parameters pf(4);
  pf << 550, 550, 319.5, 239.5;
  tdp::Cameraf cf(pf);

  tdp::ManagedHostImage<tdp::Vector3fda> pc(10000);
  for (size_t i=0; i<10000; ++i) {
    pc[i] = 0.5*tdp::Vector3fda::Random();
  }
  std::cout << std::endl;

  pangolin::CreateWindowAndBind( "GuiBase", 1200, 800);
  // Assume packed OpenGL data unless otherwise specified
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  pangolin::View& container = pangolin::Display("container");
  container.SetLayout(pangolin::LayoutEqual)
    .SetBounds(0., 1.0, 0, 1.0);
  tdp::QuickView view(640,480);
  container.AddDisplay(view);
  
  tdp::ProjectiveAssociation<4,tdp::Cameraf> projAssoc(cf, 640, 480);

  tdp::SE3f T_cw(tdp::SO3f(),tdp::Vector3fda(0,0,0));

  while(!pangolin::ShouldQuit())
  {
    projAssoc.Associate(pc, T_cw, 0, 2);

    tdp::ManagedHostImage<uint32_t> z(640,480);
    z.Fill(0);

//    projAssoc.GetAssoc(z);
    projAssoc.tex_.Download(z.ptr_, GL_RGBA, GL_UNSIGNED_BYTE);

    for (size_t i=0; i<z.Area(); ++i) {
      if (z[i] > 0) {
        int a = int((z[i]&0xFF000000)>>24);
        int r = int((z[i]&0x00FF0000)>>16);
        int g = int((z[i]&0x0000FF00)>>8) ;
        int b = int((z[i]&0x000000FF)) ;
        std::cout << std::bitset<32>(z[i]) 
          << " " << a << " " << r 
          << " " << g << " " << b 
          << " " << (z[i] & 0x00FFFFFF) << std::endl;
      }
    }

    glDisable(GL_DEPTH_TEST);

    view.Activate();
    view.glRenderTexture(projAssoc.tex_);

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    pangolin::FinishFrame();
  }

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
