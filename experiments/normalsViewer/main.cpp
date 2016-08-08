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
                  << " " << si.PixFormat().format << " (pitch: " << si.Pitch() << " bytes)" << std::endl;
        if (si.PixFormat().format.compare(pangolin::VideoFormatFromString("GRAY16LE").format)==0) {
          iD = s;
        }
        if (si.PixFormat().format.compare(pangolin::VideoFormatFromString("RGB24").format)==0) {
          iRGB = s;
        }
    }
    size_t w = video.Streams()[iRGB].Width();
    size_t h = video.Streams()[iRGB].Height();
    size_t wc = w+w%64; // for convolution
    size_t hc = h+h%64;
    float f = 550;
    float uc = (w-1.)/2.;
    float vc = (h-1.)/2.;

    // Check if video supports VideoPlaybackInterface
    pangolin::VideoPlaybackInterface* video_playback = video.Cast<pangolin::VideoPlaybackInterface>();
    const int total_frames = video_playback ? video_playback->GetTotalFrames() : std::numeric_limits<int>::max();


    std::vector<unsigned char> buffer;
    buffer.resize(video.SizeBytes()+1);

    // Create OpenGL window - guess sensible dimensions
    int menue_w = 180;
    pangolin::CreateWindowAndBind( "VideoViewer",
        (int)(video.Width() * num_streams)+menue_w,
        (int)(video.Height())
    );

    // Assume packed OpenGL data unless otherwise specified
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Setup resizable views for video streams
    std::vector<pangolin::GlPixFormat> glfmt;
    std::vector<std::pair<float,float> > gloffsetscale;
    std::vector<size_t> strides;
    std::vector<pangolin::ImageViewHandler> handlers;
    handlers.reserve(num_streams+10);

    size_t scratch_buffer_bytes = 0;

    pangolin::View& container = pangolin::Display("streams");
    container.SetLayout(pangolin::LayoutEqual)
             .SetBounds(0., 1.0, pangolin::Attach::Pix(menue_w), 1.0);
    for(unsigned int d=0; d < num_streams; ++d) {
        const pangolin::StreamInfo& si = video.Streams()[d];
        pangolin::View& view = pangolin::CreateDisplay().SetAspect(si.Aspect());
        container.AddDisplay(view);
        glfmt.push_back(pangolin::GlPixFormat(si.PixFormat()));
        gloffsetscale.push_back(std::pair<float,float>(0.0f, 1.0f) );
        if( si.PixFormat().bpp % 8 ) {
            pango_print_warn("Stream %i: Unable to display formats that are not a multiple of 8 bits.", d);
        }
        if( (8*si.Pitch()) % si.PixFormat().bpp ) {
            pango_print_warn("Stream %i: Unable to display formats whose pitch is not a whole number of pixels.", d);
        }
        if(glfmt.back().gltype == GL_DOUBLE) {
            scratch_buffer_bytes = std::max(scratch_buffer_bytes, sizeof(float)*si.Width() * si.Height());
        }
        strides.push_back( (8*si.Pitch()) / si.PixFormat().bpp );
        handlers.push_back( pangolin::ImageViewHandler(si.Width(), si.Height()) );
        view.SetHandler(&handlers.back());
    }


    tdp::QuickView viewDebugA(wc,hc);
    //pangolin::View& viewDebugA = pangolin::CreateDisplay().SetAspect(video.Streams()[iD].Aspect());
    //container.AddDisplay(viewDebugA);
    //glfmt.push_back(pangolin::GlPixFormat(pangolin::VideoFormatFromString("GRAY32F")));
    //gloffsetscale.push_back(std::pair<float,float>(0.0f, 1.0f));
    //strides.push_back(wc);
    //handlers.push_back( pangolin::ImageViewHandler(wc,hc) );
    //viewDebugA.SetHandler(&handlers.back());

    pangolin::View& viewDebugB = pangolin::CreateDisplay().SetAspect(video.Streams()[iD].Aspect());
    container.AddDisplay(viewDebugB);
    glfmt.push_back(pangolin::GlPixFormat(pangolin::VideoFormatFromString("GRAY32F")));
    gloffsetscale.push_back(std::pair<float,float>(0.0f, 1.0f));
    strides.push_back(wc);
    handlers.push_back( pangolin::ImageViewHandler(wc,hc) );
    viewDebugB.SetHandler(&handlers.back());

    pangolin::View& viewN2D = pangolin::CreateDisplay().SetAspect(video.Streams()[iD].Aspect());
    container.AddDisplay(viewN2D);
    glfmt.push_back(pangolin::GlPixFormat(pangolin::VideoFormatFromString("RGB24")));
    gloffsetscale.push_back(std::pair<float,float>(0.0f, 1.0f));
    strides.push_back(wc);
    handlers.push_back( pangolin::ImageViewHandler(wc,hc) );
    viewN2D.SetHandler(&handlers.back());

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisY)
        );
    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
      .SetHandler(new pangolin::Handler3D(s_cam));
    container.AddDisplay(d_cam);

    // current frame in memory buffer and displaying.
    pangolin::Var<int> frame("ui.frame", -1, 0, total_frames-1 );
    pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menue_w));


    pangolin::Var<int>  record_timelapse_frame_skip("viewer.record_timelapse_frame_skip", 1 );
    pangolin::Var<int>  end_frame("viewer.end_frame", std::numeric_limits<int>::max() );
    pangolin::Var<bool> video_wait("video.wait", true);
    pangolin::Var<bool> video_newest("video.newest", false);

    pangolin::Var<bool> show2DNormals("ui.show 2D Normals",false,true);
    pangolin::Var<bool> compute3Dgrads("ui.compute3Dgrads",false,true);
    pangolin::Var<bool> evaluatePlaneFit("ui.evalPlaneFit",true,true);

    if( video_playback ) {
        if(total_frames < std::numeric_limits<int>::max() ) {
            std::cout << "Video length: " << total_frames << " frames" << std::endl;
        }
        end_frame = 0;
    }
    end_frame = std::numeric_limits<int>::max();

    std::vector<unsigned char> scratch_buffer;
    scratch_buffer.resize(scratch_buffer_bytes);

    std::vector<pangolin::Image<unsigned char> > images;

#ifdef CALLEE_HAS_CPP11
    const int FRAME_SKIP = 30;
    const char show_hide_keys[]  = {'1','2','3','4','5','6','7','8','9'};
    const char screenshot_keys[] = {'!','"','#','$','%','^','&','*','('};

    // Show/hide streams
    for(size_t v=0; v < container.NumChildren() && v < 9; v++) {
        pangolin::RegisterKeyPressCallback(show_hide_keys[v], [v,&container](){
            container[v].ToggleShow();
        } );
        pangolin::RegisterKeyPressCallback(screenshot_keys[v], [v,&images,&video](){
            if(v < images.size() && images[v].ptr) {
                try{
                    pangolin::SaveImage(
                        images[v], video.Streams()[v].PixFormat(),
                        pangolin::MakeUniqueFilename("capture.png")
                    );
                }catch(std::exception e){
                    pango_print_error("Unable to save frame: %s\n", e.what());
                }
            }
        } );
    }

    pangolin::RegisterKeyPressCallback('r', [&](){
        if(!video.IsRecording()) {
            video.SetTimelapse( static_cast<size_t>(record_timelapse_frame_skip) );
            video.Record();
            pango_print_info("Started Recording.\n");
        }else{
            video.Stop();
            pango_print_info("Finished recording.\n");
        }
        fflush(stdout);
    });
    pangolin::RegisterKeyPressCallback('p', [&](){
        video.Play();
        end_frame = std::numeric_limits<int>::max();
        pango_print_info("Playing from file log.\n");
        fflush(stdout);
    });
    pangolin::RegisterKeyPressCallback('s', [&](){
        video.Source();
        end_frame = std::numeric_limits<int>::max();
        pango_print_info("Playing from source input.\n");
        fflush(stdout);
    });
    pangolin::RegisterKeyPressCallback(' ', [&](){
        end_frame = (frame < end_frame) ? frame : std::numeric_limits<int>::max();
    });
    pangolin::RegisterKeyPressCallback('w', [&](){
        video_wait = !video_wait;
        if(video_wait) {
            pango_print_info("Gui wait's for video frame.\n");
        }else{
            pango_print_info("Gui doesn't wait for video frame.\n");
        }
    });
    pangolin::RegisterKeyPressCallback('d', [&](){
        video_newest = !video_newest;
        if(video_newest) {
            pango_print_info("Discarding old frames.\n");
        }else{
            pango_print_info("Not discarding old frames.\n");
        }
    });
    pangolin::RegisterKeyPressCallback('<', [&](){
        if(video_playback) {
            frame = video_playback->Seek(frame - FRAME_SKIP) -1;
            end_frame = frame + 1;
        }else{
            pango_print_warn("Unable to skip backward.");
        }
    });
    pangolin::RegisterKeyPressCallback('>', [&](){
        if(video_playback) {
            frame = video_playback->Seek(frame + FRAME_SKIP) -1;
            end_frame = frame + 1;
        }else{
            end_frame = frame + FRAME_SKIP;
        }
    });
    pangolin::RegisterKeyPressCallback(',', [&](){
        if(video_playback) {
            frame = video_playback->Seek(frame - 1) -1;
            end_frame = frame+1;
        }else{
            pango_print_warn("Unable to skip backward.");
        }
    });
    pangolin::RegisterKeyPressCallback('.', [&](){
        // Pause at next frame
        end_frame = frame+1;
    });
    pangolin::RegisterKeyPressCallback('0', [&](){
        video.RecordOneFrame();
    });
    pangolin::RegisterKeyPressCallback('a', [&](){
        // Adapt scale
        for(unsigned int i=0; i<images.size(); ++i) {
            if(container[i].HasFocus()) {
                pangolin::Image<unsigned char>& img = images[i];
                pangolin::ImageViewHandler& ivh = handlers[i];

                const bool have_selection = std::isfinite(ivh.GetSelection().Area()) && std::abs(ivh.GetSelection().Area()) >= 4;
                pangolin::XYRangef froi = have_selection ? ivh.GetSelection() : ivh.GetViewToRender();
                gloffsetscale[i] = pangolin::GetOffsetScale(img, froi.Cast<int>(), glfmt[i]);
            }
        }
    });
    pangolin::RegisterKeyPressCallback('g', [&](){
        std::pair<float,float> os_default(0.0f, 1.0f);

        // Get the scale and offset from the container that has focus.
        for(unsigned int i=0; i<images.size(); ++i) {
            if(container[i].HasFocus()) {
                pangolin::Image<unsigned char>& img = images[i];
                pangolin::ImageViewHandler& ivh = handlers[i];

                const bool have_selection = std::isfinite(ivh.GetSelection().Area()) && std::abs(ivh.GetSelection().Area()) >= 4;
                pangolin::XYRangef froi = have_selection ? ivh.GetSelection() : ivh.GetViewToRender();
                os_default = pangolin::GetOffsetScale(img, froi.Cast<int>(), glfmt[i]);
                break;
            }
        }

        // Adapt scale for all images equally
        // TODO : we're assuming the type of all the containers images' are the same.
        for(unsigned int i=0; i<images.size(); ++i) {
            gloffsetscale[i] = os_default;
        }

    });
#endif // CALLEE_HAS_CPP11

    tdp::ManagedHostImage<float> d(wc, hc);
    tdp::ManagedHostImage<float> debugA(wc, hc);
    tdp::ManagedHostImage<float> debugB(wc, hc);
    tdp::ManagedHostImage<Eigen::Matrix<uint8_t,3,1>> n2D(wc,hc);
    memset(n2D.ptr_,0,n2D.SizeBytes());
    tdp::ManagedHostImage<Eigen::Vector3f> n2Df(wc,hc);
    tdp::ManagedHostImage<Eigen::Vector3f> n(wc,hc);

    pangolin::GlBufferCudaPtr cuNbuf(pangolin::GlArrayBuffer, wc*hc, GL_FLOAT, 3, cudaGraphicsMapFlagsNone, GL_DYNAMIC_DRAW);

    tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
    tdp::ManagedDeviceImage<float> cuD(wc, hc);
    tdp::ManagedDeviceImage<float> cuDu(wc, hc);
    tdp::ManagedDeviceImage<float> cuDv(wc, hc);
    tdp::ManagedDeviceImage<float> cuTmp(wc, hc);
    // Stream and display video
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
                //pangolin::Image<unsigned char>debugARaw(wc,hc,wc*sizeof(float),(uint8_t*)debugA.ptr);
                //images.push_back(debugARaw);
                pangolin::Image<unsigned char>debugBRaw(wc,hc,wc*sizeof(float),(uint8_t*)debugB.ptr_);
                images.push_back(debugBRaw);
                pangolin::Image<unsigned char>n2DRaw(wc,hc,wc*3,(uint8_t*)n2D.ptr_);
                images.push_back(n2DRaw);
            }
        }

        pangolin::basetime t0 = pangolin::TimeNow();

        tdp::Image<uint16_t> dRaw(images[iD].w, images[iD].h,
            images[iD].pitch, reinterpret_cast<uint16_t*>(images[iD].ptr));
        CopyImage(dRaw, cuDraw, cudaMemcpyHostToDevice);
        ConvertDepth(cuDraw, cuD, 1e-4);

        cudaDeviceSynchronize();
        pangolin::basetime tDepth = pangolin::TimeNow();

        // upload to gpu
        // get derivatives using shar kernel
        float kernelA[3] = {1,0,-1};
        setConvolutionKernel(kernelA);
        convolutionRowsGPU((float*)cuTmp.ptr_,(float*)cuD.ptr_,wc,hc);
        float kernelB[3] = {3/32.,10/32.,3/32.};
        setConvolutionKernel(kernelB);
        convolutionColumnsGPU((float*)cuDu.ptr_,(float*)cuTmp.ptr_,wc,hc);
        convolutionRowsGPU((float*)cuTmp.ptr_,(float*)cuD.ptr_,wc,hc);
        setConvolutionKernel(kernelA);
        convolutionColumnsGPU((float*)cuDv.ptr_,(float*)cuTmp.ptr_,wc,hc);

        cudaDeviceSynchronize();
        pangolin::basetime tGrad = pangolin::TimeNow();

        {
          pangolin::CudaScopedMappedPtr cuNbufp(cuNbuf);
          cudaMemset(*cuNbufp,0, hc*wc*sizeof(Eigen::Vector3f));
          tdp::Image<Eigen::Vector3f> cuN(wc, hc, wc*sizeof(Eigen::Vector3f), (Eigen::Vector3f*)*cuNbufp);
          ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
          if (show2DNormals) {
            CopyImage(cuN, n2Df, cudaMemcpyDeviceToHost);
            //ConvertPixelsMultiChannel<uint8_t,float,3>(n2D,n2Df,128,127);
            for(size_t y=0; y < n2Df.h_; ++y) {
              for(size_t x=0; x < n2Df.w_*3; ++x) {
                ((uint8_t*)n2D.RowPtr((int)y))[x] = floor(((float*)n2Df.RowPtr((int)y))[x]*128+127);
              }
            }
          }
        }
        cudaDeviceSynchronize();
        pangolin::basetime tNormal = pangolin::TimeNow();

        std::cout << pangolin::TimeDiff_s(t0,tDepth) << "\t"
          << pangolin::TimeDiff_s(tDepth,tGrad) << "\t"
          << pangolin::TimeDiff_s(tGrad,tNormal) << "\t"
          << pangolin::TimeDiff_s(t0,tNormal) << "\t"<< std::endl;

        if (evaluatePlaneFit) {

        }

        glEnable(GL_DEPTH_TEST);
        d_cam.Activate(s_cam);
        pangolin::glDrawAxis(1);
        pangolin::RenderVbo(cuNbuf);

        glLineWidth(1.5f);
        glDisable(GL_DEPTH_TEST);

        CopyImage(cuDu, debugA, cudaMemcpyDeviceToHost);
        viewDebugA.SetImage(debugA);
        CopyImage(cuDv, debugB, cudaMemcpyDeviceToHost);
        for(unsigned int i=0; i<images.size(); ++i)
        {
            if(container[i].IsShown()) {
                container[i].Activate();
                pangolin::Image<unsigned char>& image = images[i];

                // Get texture of correct dimension / format
                const pangolin::GlPixFormat& fmt = glfmt[i];
                pangolin::GlTexture& tex = pangolin::TextureCache::I().GlTex((GLsizei)image.w, (GLsizei)image.h, fmt.scalable_internal_format, fmt.glformat, GL_FLOAT);

                // Upload image data to texture
                tex.Bind();
                glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)strides[i]);
                tex.Upload(image.ptr,0,0, (GLsizei)image.w, (GLsizei)image.h, fmt.glformat, fmt.gltype);

                // Render
                handlers[i].UpdateView();
                handlers[i].glSetViewOrtho();
                const std::pair<float,float> os = gloffsetscale[i];
                pangolin::GlSlUtilities::OffsetAndScale(os.first, os.second);
                handlers[i].glRenderTexture(tex);
                pangolin::GlSlUtilities::UseNone();
                handlers[i].glRenderOverlay();
            }
        }
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

        // leave in pixel orthographic for slider to render.
        pangolin::DisplayBase().ActivatePixelOrthographic();
        if(video.IsRecording()) {
            pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f, pangolin::DisplayBase().v.h-14.0f, 7.0f);
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
