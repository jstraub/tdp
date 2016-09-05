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

#include <tdp/depth.h>
#include <tdp/volume.h>
#include <tdp/managed_volume.h>
#include <tdp/image.h>
#include <tdp/hist.h>
#include <tdp/manifold/SE3.h>
#include <tdp/calibration/planeEstimation.h>

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


    GUI gui(1200, 800, video );

    size_t w = video.Streams()[gui.iRGB].Width();
    size_t h = video.Streams()[gui.iRGB].Height();

    size_t wc = w;//+w%64; // for convolution
    size_t hc = h;//+h%64;

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
        pangolin::ModelViewLookAt(0,0.5,-3, 0,0,0, pangolin::AxisY)
        );
    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
      .SetHandler(new pangolin::Handler3D(s_cam));
    gui.container().AddDisplay(d_cam);

    tdp::ManagedHostImage<float> debugA(wc, hc);
    tdp::ManagedHostImage<float> debugB(wc, hc);
    tdp::QuickView viewDebugA(wc,hc);
    gui.container().AddDisplay(viewDebugA);
    tdp::QuickView viewDebugB(wc,hc);
    gui.container().AddDisplay(viewDebugB);
    pangolin::DataLog logStats;
    pangolin::Plotter plotStats(&logStats, 0.f,100.f, 0.f,1.f, 10.f, 0.1f);
    gui.container().AddDisplay(plotStats);

    tdp::ManagedHostImage<float> d(wc, hc);
    tdp::ManagedDeviceImage<uint16_t> cuDraw(w, h);
    tdp::ManagedDeviceImage<float> cuD(wc, hc);
    tdp::ManagedDeviceImage<float> cuRho(wc, hc);
    tdp::ManagedDeviceImage<float> cuPlaneDeriv(wc, hc);

    tdp::SE3<float> T_wd(Eigen::Matrix4f::Identity());
    tdp::Camera<float> camD(Eigen::Vector4f(550,550,319.5,239.5)); 

    pangolin::Var<float> huberAlpha("ui.alpha", 0.3, 0., 1.);
    pangolin::Var<int> planeEstNumIter("ui.# iter", 100, 1, 1000);
    pangolin::Var<float> planeEstEps("ui.10^eps", -6, -10, 0);
    pangolin::Var<float> histDx("ui.Hist dx", 0.01, 1e-3, 0.01);
    pangolin::Var<float> sensorScale("ui.sensor scale", 1e-3, 1e-4, 1e-3);

    // Stream and display video
    //
    while(!pangolin::ShouldQuit())
    {
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 1.0f, 1.0f);

        gui.NextFrames();

        tdp::Image<uint16_t> dRaw;
        if (!gui.ImageD(dRaw)) continue;
        CopyImage(dRaw, cuDraw, cudaMemcpyHostToDevice);
        ConvertDepthGpu(cuDraw, cuD, sensorScale, 0.1, 4.0);

        Eigen::Vector3f nd(0,0,-1);
        tdp::PlaneEstimation planeEstimator(&cuD,camD,huberAlpha.Get());
        //planeEstimator.Reset(&cuD,huberAlpha.Get());
        planeEstimator.Compute(nd, pow(10,planeEstEps), planeEstNumIter.Get());
        nd = planeEstimator.GetMinimum();
        std::cout << "n=" << nd.transpose()/nd.norm() 
          << " d=" << 1./nd.norm() << std::endl;
        CopyImage(planeEstimator.cuF_,debugA,cudaMemcpyDeviceToHost);

        tdp::Hist hist;
        hist.Compute(debugA, histDx, true);
        logStats.Clear();
        for (size_t i=0; i<hist.hist_.size(); ++i) 
          logStats.Log(hist.hist_[i]);
        //plotStats.Scroll(100,0);

        glEnable(GL_DEPTH_TEST);
        d_cam.Activate(s_cam);
        pangolin::glDrawAxis(1);
        //pangolin::RenderVbo(cuNbuf);

        glLineWidth(1.5f);
        glDisable(GL_DEPTH_TEST);

        gui.ShowFrames();
        viewDebugA.SetImage(debugA);

        ConvertDepthToInverseDepthGpu(cuDraw, cuRho, sensorScale, 0.1, 4.0);
        debugB.CopyFrom(cuRho,cudaMemcpyDeviceToHost);
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
