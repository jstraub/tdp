/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
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
#include <tdp/managed_image.h>

#include <tdp/depth.h>
#include <tdp/pc.h>
#include <tdp/camera.h>
#include <tdp/quickView.h>
#include <tdp/eigen/dense.h>
#ifdef CUDA_FOUND
#include <tdp/normals.h>
#endif

#include "gui.hpp"

void VideoViewer(const std::string& input_uri, const std::string& output_uri)
{

  // Open Video by URI
  pangolin::VideoRecordRepeat video(input_uri, output_uri);
  const size_t num_streams = video.Streams().size();

  if(num_streams == 0) {
    pango_print_error("No video streams from device.\n");
    return;
  }

  GUI gui(1200,800,video);

  size_t w = video.Streams()[gui.iD].Width();
  size_t h = video.Streams()[gui.iD].Height();
  size_t wc = w;
  size_t hc = h;
  float f = 550;
  float uc = (w-1.)/2.;
  float vc = (h-1.)/2.;

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisNegY)
      );
  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetHandler(new pangolin::Handler3D(s_cam));
  gui.container().AddDisplay(d_cam);

  tdp::ManagedHostImage<float> d(wc, hc);
  tdp::ManagedHostImage<tdp::Vector3fda> pc(wc, hc);
  tdp::ManagedHostImage<float> debugA(wc, hc);
  tdp::ManagedHostImage<float> debugB(wc, hc);

  pangolin::GlBuffer pcIbo;
  pangolin::MakeTriangleStripIboForVbo(pcIbo,wc,hc);
  pangolin::GlBuffer pcVbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer pcNbo(pangolin::GlArrayBuffer,wc*hc,GL_FLOAT,3);
  pangolin::GlBuffer pcCbo(pangolin::GlArrayBuffer,wc*hc,GL_UNSIGNED_BYTE,3);

  tdp::Camera<float> cam(Eigen::Vector4f(550,550,319.5,239.5)); 

  pangolin::GlSlProgram matcap;
  matcap.AddShaderFromFile(pangolin::GlSlVertexShader,
      "/home/jstraub/workspace/research/tdp/shaders/matcap.vert");
  matcap.AddShaderFromFile(pangolin::GlSlFragmentShader,
      "/home/jstraub/workspace/research/tdp/shaders/matcap.frag");
  matcap.Link();
  pangolin::GlSlProgram colorPc;
  colorPc.AddShaderFromFile(pangolin::GlSlVertexShader,
      "/home/jstraub/workspace/research/tdp/shaders/colorPc.vert");
  colorPc.AddShaderFromFile(pangolin::GlSlFragmentShader,
      "/home/jstraub/workspace/research/tdp/shaders/colorPc.frag");
  colorPc.Link();

  // http://www.geeks3d.com/20110908/opengl-3-3-sampler-objects-control-your-texture-units/
  GLuint linearFiltering;
  glGenSamplers(1,&linearFiltering);
  glSamplerParameteri(linearFiltering, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glSamplerParameteri(linearFiltering, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glSamplerParameteri(linearFiltering, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  pangolin::GlTexture matcapTex(512,512,GL_RGB8);
  pangolin::TypedImage matcapImg = pangolin::LoadImage(
      "/home/jstraub/workspace/research/tdp/shaders/normal.jpg");
  matcapTex.Upload(matcapImg.ptr,GL_RGB,GL_UNSIGNED_BYTE);

#ifdef CUDA_FOUND
  tdp::ManagedDeviceImage<float> cuD(wc, hc);
  tdp::ManagedDeviceImage<tdp::Vector3fda> cuN(wc, hc);
#endif
  tdp::ManagedHostImage<tdp::Vector3fda> n(wc, hc);
  for (size_t i=0; i<wc*hc; ++i)
    n[i] = tdp::Vector3fda(0.,0.,1.);

  // Stream and display video
  while(!pangolin::ShouldQuit())
  {
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);

    gui.NextFrames();

    tdp::Image<tdp::Vector3bda> rgb;
    if (!gui.ImageRGB(rgb)) continue;
    tdp::Image<uint16_t> dRaw;
    if (!gui.ImageD(dRaw)) continue;
    tdp::ConvertDepth(dRaw, d, 1e-3, 0.1, 4.);
    tdp::Depth2PC(d,cam,pc);
#ifdef CUDA_FOUND
    // compute normals if we have CUDA (normal extraction not
    // implemented on CPU (yet))
    cuD.CopyFrom(d,cudaMemcpyHostToDevice);
    tdp::Depth2Normals(cuD, cam, cuN);
    n.CopyFrom(cuN,cudaMemcpyDeviceToHost);
#endif

    glEnable(GL_DEPTH_TEST);
    d_cam.Activate(s_cam);
    pangolin::glDrawAxis(0.1);
    Eigen::Matrix3f Kinv = cam.GetKinv();
    pangolin::glDrawFrustrum(Kinv, wc, hc, 0.1);

    pangolin::OpenGlMatrix P = s_cam.GetProjectionMatrix();
    pangolin::OpenGlMatrix MV = s_cam.GetModelViewMatrix();
    if (gui.useMatCap) {
      pcVbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
      pcNbo.Bind();
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0); 

      glEnableVertexAttribArray(0);                                               
      glEnableVertexAttribArray(1);                                               
      pcVbo.Upload(pc.ptr_,pc.SizeBytes(),0);
      pcNbo.Upload(n.ptr_,n.SizeBytes(),0);
      matcap.Bind();
      //TODO: matcap.SetUniform("matcap",matcapTex);
      // https://www.opengl.org/wiki/Sampler_(GLSL)#Binding_textures_to_samplers
      //glActiveTexture(GL_TEXTURE0 + 2);
      //matcapTex.Bind();
      //glBindSampler(2,linearFiltering);
      //matcap.SetUniform("matcap",2);
      matcap.SetUniform("P",P);
      matcap.SetUniform("MV",MV);
    } else {
      pcVbo.Bind();
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); 
      pcCbo.Bind();
      glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0); 

      glEnableVertexAttribArray(0);                                               
      glEnableVertexAttribArray(1);                                               
      pcVbo.Upload(pc.ptr_,pc.SizeBytes(),0);
      pcCbo.Upload(rgb.ptr_,rgb.SizeBytes(),0);
      colorPc.Bind();
      colorPc.SetUniform("P",P);
      colorPc.SetUniform("MV",MV);
    }

    pcIbo.Bind();
    glDrawElements(GL_TRIANGLE_STRIP,pcIbo.num_elements, pcIbo.datatype, 0);
    pcIbo.Unbind();

    if (gui.useMatCap) {
      //glBindSampler(2,0);
      matcapTex.Unbind();
      matcap.Unbind();
      glDisableVertexAttribArray(1);
      glDisableVertexAttribArray(0);
      pcNbo.Unbind();
      pcVbo.Unbind();
    } else {
      colorPc.Unbind();
      glDisableVertexAttribArray(1);
      glDisableVertexAttribArray(0);
      pcCbo.Unbind();
      pcVbo.Unbind();
    }
    //glDisableClientState(GL_COLOR_ARRAY);
    //cbo.Unbind();

    glLineWidth(1.5f);
    glDisable(GL_DEPTH_TEST);
    gui.ShowFrames();

    // leave in pixel orthographic for slider to render.
    pangolin::DisplayBase().ActivatePixelOrthographic();
    if(video.IsRecording()) {
      pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f,
          pangolin::DisplayBase().v.h-14.0f, 7.0f);
    }
    pangolin::FinishFrame();
  }
  glDeleteSamplers(1,&linearFiltering);
}


int main( int argc, char* argv[] )
{
  const std::string dflt_output_uri = "pango://video.pango";

  if( argc > 1 ) {
    const std::string input_uri = std::string(argv[1]);
    const std::string output_uri = (argc > 2) ? std::string(argv[2]) : dflt_output_uri;
    try{
      VideoViewer(input_uri, output_uri);
    } catch (pangolin::VideoException e) {
      std::cout << e.what() << std::endl;
    }
  }else{
    const std::string input_uris[] = {
      "dc1394:[fps=30,dma=10,size=640x480,iso=400]//0",
      "convert:[fmt=RGB24]//v4l:///dev/video0",
      "convert:[fmt=RGB24]//v4l:///dev/video1",
      "openni:[img1=rgb]//",
      "test:[size=160x120,n=1,fmt=RGB24]//"
        ""
    };

    std::cout << "Usage  : VideoViewer [video-uri]" << std::endl << std::endl;
    std::cout << "Where video-uri describes a stream or file resource, e.g." << std::endl;
    std::cout << "\tfile:[realtime=1]///home/user/video/movie.pvn" << std::endl;
    std::cout << "\tfile:///home/user/video/movie.avi" << std::endl;
    std::cout << "\tfiles:///home/user/seqiemce/foo%03d.jpeg" << std::endl;
    std::cout << "\tdc1394:[fmt=RGB24,size=640x480,fps=30,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tdc1394:[fmt=FORMAT7_1,size=640x480,pos=2+2,iso=400,dma=10]//0" << std::endl;
    std::cout << "\tv4l:///dev/video0" << std::endl;
    std::cout << "\tconvert:[fmt=RGB24]//v4l:///dev/video0" << std::endl;
    std::cout << "\tmjpeg://http://127.0.0.1/?action=stream" << std::endl;
    std::cout << "\topenni:[img1=rgb]//" << std::endl;
    std::cout << std::endl;

    // Try to open some video device
    for(int i=0; !input_uris[i].empty(); ++i )
    {
      try{
        pango_print_info("Trying: %s\n", input_uris[i].c_str());
        VideoViewer(input_uris[i], dflt_output_uri);
        return 0;
      }catch(pangolin::VideoException) { }
    }
  }

  return 0;
}
