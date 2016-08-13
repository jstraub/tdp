#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glcuda.h>

#include <Eigen/Dense>
#include <tdp/managed_image.h>

#include <tdp/convolutionSeparable.h>
#include <tdp/depth.h>
#include <tdp/normals.h>
#include <tdp/quickView.h>
#include <tdp/volume.h>
#include <tdp/managed_volume.h>
#include <tdp/image.h>
#include <tdp/manifold/SE3.h>
#include <tdp/tsdf.h>
#include <tdp/nvidia/helper_cuda.h>

template<typename To, typename From>
void ConvertPixels(pangolin::Image<To>& to, const pangolin::Image<From>& from, float scale, float offset)
{
  memset(to.ptr,0,to.SizeBytes());
  for(size_t y=0; y < from.h; ++y) {
    for(size_t x=0; x < from.w; ++x) {
      to.RowPtr((int)y)[x] = static_cast<To>(static_cast<float>(from.RowPtr((int)y)[x])*scale+offset);
    }
  }
}

//template<typename To, typename From, int D>
//void ConvertPixelsMultiChannel(pangolin::Image<Eigen::Matrix<To,D,1>>& to, const pangolin::Image<Eigen::Matrix<From,D,1>>& from, float scale, float offset)
//{
//  memset(to.ptr,0,to.SizeBytes());
//  for(size_t y=0; y < from.h; ++y) {
//    for(size_t x=0; x < from.w; ++x) {
//      to.RowPtr((int)y)[x] = ((from.RowPtr((int)y)[x]).cast<float>()*scale+offset).cast<To>();
//    }
//  }
//}

void VideoViewer(const std::string& input_uri, const std::string& output_uri)
{
    // Open Video by URI
    pangolin::VideoRecordRepeat video(input_uri, output_uri);
    const size_t num_streams = video.Streams().size();
    if(num_streams == 0) {
        pango_print_error("No video streams from device.\n");
        return;
    }

    // Output details of video stream
    size_t iRGB, iD;
    for(size_t s = 0; s < num_streams; ++s) {
        const pangolin::StreamInfo& si = video.Streams()[s];
        std::cout << "Stream " << s << ": " << si.Width() << " x " << si.Height()
                  << " " << si.PixFormat().format 
                  << " (pitch: " << si.Pitch() << " bytes)" << std::endl;
        if (si.PixFormat().format.compare(
              pangolin::VideoFormatFromString("GRAY16LE").format)==0) {
          iD = s;
        }
        if (si.PixFormat().format.compare(
              pangolin::VideoFormatFromString("RGB24").format)==0) {
          iRGB = s;
        }
    }
    size_t w = video.Streams()[iRGB].Width();
    size_t h = video.Streams()[iRGB].Height();
    size_t wc = w;//+w%64; // for convolution
    size_t hc = h;//+h%64;
    float f = 550;
    float uc = (w-1.)/2.;
    float vc = (h-1.)/2.;

    size_t dTSDF = 64;
    size_t wTSDF = wc;
    size_t hTSDF = hc;

    // Check if video supports VideoPlaybackInterface
    pangolin::VideoPlaybackInterface* video_playback = 
      video.Cast<pangolin::VideoPlaybackInterface>();
    const int total_frames = video_playback ? video_playback->GetTotalFrames() : std::numeric_limits<int>::max();

    std::vector<unsigned char> buffer;
    buffer.resize(video.SizeBytes()+1);

    GUI gui((video.Width() * num_streams)+menue_w, video.Height() );

    if( video_playback ) {
        if(total_frames < std::numeric_limits<int>::max() ) {
            std::cout << "Video length: " << total_frames << " frames" << std::endl;
        }
        end_frame = 0;
    }
    end_frame = std::numeric_limits<int>::max();

    std::vector<pangolin::Image<unsigned char> > images;

    tdp::ManagedHostImage<float> d(wc, hc);
    tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
    tdp::ManagedDeviceImage<float> cuD(wc, hc);
    tdp::ManagedDeviceImage<float> cuPlaneDeriv(wc, hc);

    tdp::SE3<float> T_wd(Eigen::Matrix4f::Identity());
    tdp::Camera<float> camD(Eigen::Vector4f(550,550,319.5,239.5)); 

    tdp::ManagedHostImage<float> debugA(wc, hc);
    tdp::ManagedHostImage<float> debugB(wc, hc);
    tdp::QuickView viewDebugA(wc,hc);
    gui.container.AddDisplay(viewDebugA);
    tdp::QuickView viewDebugB(wc,hc);
    gui.container.AddDisplay(viewDebugB);
    // Stream and display video
    //
    PlaneEstimation planeEstimator(cuD,cam);
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 1.0f, 1.0f);

        if(frame.GuiChanged()) {
            if(video_playback) {
                frame = video_playback->Seek(frame) -1;
            }
            end_frame = frame + 1;
        }
        if ( frame < end_frame ) {
            if( video.Grab(&buffer[0], images, video_wait, video_newest) ) {
                frame = frame +1;
            }
        }

        pangolin::basetime t0 = pangolin::TimeNow();

        tdp::Image<uint16_t> dRaw(images[iD].w, images[iD].h,
            images[iD].pitch, reinterpret_cast<uint16_t*>(images[iD].ptr));
        CopyImage(dRaw, cuDraw, cudaMemcpyHostToDevice);
        ConvertDepth(cuDraw, cuD, 1e-4);

        cudaDeviceSynchronize();
        pangolin::basetime tDepth = pangolin::TimeNow();

        planeEstimator.Reset(cuD);
        Eigen::Vector3f nd(0,0,-1);
        planeEstimator.Compute(nd, 1e-6, 100);
        nd = planeEstimator.GetMinimum();
        std::cout << "n=" << nd.transpose()/nd.norm() 
          << " d=" << 1./nd.norm() << std::endl;
        CopyImage(planeEstimator.cuF_,debugA,cudaMemcpyDeviceToHost);

        std::cout << pangolin::TimeDiff_s(t0,tDepth) << "\t" << std::endl;

        glEnable(GL_DEPTH_TEST);
        d_cam.Activate(s_cam);
        pangolin::glDrawAxis(1);
        //pangolin::RenderVbo(cuNbuf);

        glLineWidth(1.5f);
        glDisable(GL_DEPTH_TEST);

        for(unsigned int i=0; i<images.size(); ++i)
        {
          if(container[i].IsShown()) {
            pangolin::Image<unsigned char>& image = images[i];
            streamViews[i]->SetImage(image, glfmt[i], strides[i]);
          }
        }
        viewDebugA.SetImage(debugA);
        viewDebugB.SetImage(debugB);

        // leave in pixel orthographic for slider to render.
        pangolin::DisplayBase().ActivatePixelOrthographic();
        if(video.IsRecording()) {
          pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f, 
              pangolin::DisplayBase().v.h-14.0f, 7.0f);
        }
        pangolin::FinishFrame();
    }
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
