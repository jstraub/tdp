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
#include <tdp/pc.h>
#include <tdp/quickView.h>
#include <tdp/volume.h>
#include <tdp/managed_volume.h>
#include <tdp/image.h>
#include <tdp/manifold/SE3.h>
#include <tdp/icp.h>
#include <tdp/tsdf.h>
#include <tdp/pyramid.h>
#include <tdp/managed_pyramid.h>
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

    size_t dTSDF = 64;
    size_t wTSDF = wc;
    size_t hTSDF = hc;

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
    std::vector<tdp::QuickView*> streamViews; streamViews.reserve(10);
    for(unsigned int d=0; d < num_streams; ++d) {
        const pangolin::StreamInfo& si = video.Streams()[d];
        streamViews.push_back(new tdp::QuickView(si.Width(), si.Height()));
        container.AddDisplay(*streamViews.back());
        glfmt.push_back(pangolin::GlPixFormat(si.PixFormat()));
        strides.push_back( (8*si.Pitch()) / si.PixFormat().bpp );
    }


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
    pangolin::Var<int>  end_frame("ui.end_frame", std::numeric_limits<int>::max() );
    pangolin::Var<bool> video_wait("ui.wait", true);
    pangolin::Var<bool> video_newest("ui.newest", false);

    pangolin::Var<float> tsdfDmin("ui.d min",0.05,0.0,0.1);
    pangolin::Var<float> tsdfDmax("ui.d max",1.0,0.1,2.);
    pangolin::Var<float> tsdfRho0("ui.rho0",0.1,0.,1.);
    pangolin::Var<float> tsdfDRho("ui.d rho",0.1,0.,1.);
    pangolin::Var<float> tsdfMu("ui.mu",0.1,0.,1.);
    pangolin::Var<int> tsdfSliceD("ui.TSDF slice D",dTSDF/2,0,dTSDF-1);
    pangolin::Var<bool> resetTSDF("ui.reset TSDF", false, false);
    pangolin::Var<bool> runICP("ui.run ICP", false, false);
    pangolin::Var<float> icpAngleThr_deg("ui.icp angle thr",30,0.,180.);
    pangolin::Var<float> icpDistThr("ui.icp dist thr",0.,0.,1.);

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
#endif // CALLEE_HAS_CPP11

    tdp::ManagedHostImage<float> d(wc, hc);

    tdp::ManagedHostImage<Eigen::Matrix<uint8_t,3,1>> n2D(wc,hc);
    memset(n2D.ptr_,0,n2D.SizeBytes());
    tdp::ManagedHostImage<Eigen::Vector3f> n2Df(wc,hc);
    tdp::ManagedHostImage<Eigen::Vector3f> n(wc,hc);

    tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
    tdp::ManagedDeviceImage<float> cuD(wc, hc);


    tdp::ManagedHostVolume<float> W(wTSDF, hTSDF, dTSDF);
    tdp::ManagedHostVolume<float> TSDF(wTSDF, hTSDF, dTSDF);
    tdp::ManagedHostImage<float> dEst(wTSDF, hTSDF);
    W.Fill(0.);
    TSDF.Fill(-tsdfMu);
    dEst.Fill(0.);
    tdp::ManagedDeviceVolume<float> cuW(wTSDF, hTSDF, dTSDF);
    tdp::ManagedDeviceVolume<float> cuTSDF(wTSDF, hTSDF, dTSDF);
    tdp::ManagedDeviceImage<float> cuDEst(wTSDF, hTSDF);

    tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
    tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
    tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);

    tdp::SE3<float> T_rd(Eigen::Matrix4f::Identity());
    tdp::Camera<float> camR(Eigen::Vector4f(275,275,159.5,119.5)); 
    tdp::Camera<float> camD(Eigen::Vector4f(550,550,319.5,239.5)); 

    // ICP stuff
    tdp::ManagedHostPyramid<float,3> dPyr(wc,hc);
    tdp::ManagedHostPyramid<float,3> dPyrEst(wc,hc);
    tdp::ManagedDevicePyramid<float,3> cuDPyr(wc,hc);
    tdp::ManagedDevicePyramid<float,3> cuDPyrEst(wc,hc);
    tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_m(wc,hc);
    tdp::ManagedDevicePyramid<tdp::Vector3fda,3> pcs_c(wc,hc);
    tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_m(wc,hc);
    tdp::ManagedDevicePyramid<tdp::Vector3fda,3> ns_c(wc,hc);
    tdp::Matrix3fda R_mc = tdp::Matrix3fda::Identity();
    tdp::Vector3fda t_mc = tdp::Vector3fda::Zero();

    tdp::ManagedHostImage<float> debugA(wTSDF, hTSDF);
    tdp::ManagedHostImage<float> debugB(wTSDF, hTSDF);
    tdp::QuickView viewDebugA(wTSDF,hTSDF);
    container.AddDisplay(viewDebugA);
    tdp::QuickView viewDebugB(wTSDF,hTSDF);
    container.AddDisplay(viewDebugB);
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
            }
        }

        pangolin::basetime t0 = pangolin::TimeNow();

        tdp::Image<uint16_t> dRaw(images[iD].w, images[iD].h,
            images[iD].pitch, reinterpret_cast<uint16_t*>(images[iD].ptr));
        CopyImage(dRaw, cuDraw, cudaMemcpyHostToDevice);
        ConvertDepth(cuDraw, cuD, 1e-4);

        if (runICP) {
          // construct pyramid  
          // TODO: could also construct the pyramid directly on the GPU
          tdp::ConstructPyramidFromImage<float,3>(cuD, dPyr, cudaMemcpyDeviceToHost);
          tdp::ConstructPyramidFromImage<float,3>(cuDEst, dPyrEst, cudaMemcpyDeviceToHost);
          CopyPyramid(dPyr,cuDPyr,cudaMemcpyHostToDevice);
          CopyPyramid(dPyrEst,cuDPyrEst,cudaMemcpyHostToDevice);

          tdp::PyramidDepth2PCs(cuDPyrEst,camD,pcs_m);
          tdp::PyramidDepth2PCs(cuDPyr,camD,pcs_c);
          // TODO: dont redo the work of convolving - build pyramid of
          // ddu and ddv and compute normals from that
          tdp::Depth2Normals(cuDPyrEst,camD,ns_m);
          tdp::Depth2Normals(cuDPyr,camD,ns_c);

          R_mc = tdp::Matrix3fda::Identity();
          t_mc = tdp::Vector3fda::Zero();
          std::vector<size_t> maxIt{10,6,3};
          tdp::ICP::ComputeProjective(pcs_m, ns_m, pcs_c, ns_c, R_mc,
              t_mc, camD, maxIt, icpAngleThr_deg, icpDistThr); 
          std::cout << "R_mc" << std::endl << R_mc << std::endl 
            << "t_mc " << t_mc.transpose() << std::endl;
          T_rd.matrix().topLeftCorner<3,3>() = R_mc;
          T_rd.matrix().topRightCorner<3,1>() = t_mc;
        }
        pangolin::basetime tDepth = pangolin::TimeNow();

        if (pangolin::Pushed(resetTSDF)) {
          W.Fill(0.);
          TSDF.Fill(0.);
          dEst.Fill(0.);
          tdp::CopyImage(dEst, cuDEst, cudaMemcpyHostToDevice);
          tdp::CopyVolume(TSDF, cuTSDF, cudaMemcpyHostToDevice);
          tdp::CopyVolume(W, cuW, cudaMemcpyHostToDevice);
        }

        tsdfRho0 = 1./tsdfDmax;
        tsdfDRho = (1./tsdfDmin - tsdfRho0)/float(dTSDF-1);

        AddToTSDF(cuTSDF, cuW, cuD, T_rd, camR, camD, tsdfRho0, tsdfDRho, tsdfMu); 

        checkCudaErrors(cudaDeviceSynchronize());
        pangolin::basetime tAddTSDF = pangolin::TimeNow();

        RayTraceTSDF(cuTSDF, cuDEst, T_rd, camR, camD, tsdfRho0, tsdfDRho, tsdfMu); 

        checkCudaErrors(cudaDeviceSynchronize());
        pangolin::basetime tRayTrace = pangolin::TimeNow();

        std::cout << pangolin::TimeDiff_s(t0,tDepth) << "\t"
          << pangolin::TimeDiff_s(tDepth,tAddTSDF) << "\t"
          << pangolin::TimeDiff_s(tAddTSDF,tRayTrace) << "\t"
          << pangolin::TimeDiff_s(t0,tRayTrace) << "\t"<< std::endl;

        glEnable(GL_DEPTH_TEST);
        d_cam.Activate(s_cam);
        pangolin::glDrawAxis(1);
        //pangolin::RenderVbo(cuNbuf);

        glLineWidth(1.5f);
        glDisable(GL_DEPTH_TEST);

        CopyImage(cuDEst, debugA, cudaMemcpyDeviceToHost);
        //if (tsdfSliceD.GuiChanged()) {
        tdp::Image<float> sliceTSDF(cuTSDF.w_, cuTSDF.h_, cuTSDF.ImagePtr(std::min((int)cuTSDF.d_-1,tsdfSliceD.Get())));
        CopyImage(sliceTSDF, debugB, cudaMemcpyDeviceToHost);
        //}
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
            pangolin::glRecordGraphic(pangolin::DisplayBase().v.w-14.0f, pangolin::DisplayBase().v.h-14.0f, 7.0f);
        }
        pangolin::FinishFrame();
    }

    for (size_t i=0; i<streamViews.size(); ++i)
      delete streamViews[i];
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
