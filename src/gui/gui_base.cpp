#include <tdp/gui/gui_base.hpp>

namespace tdp {

GuiBase::GuiBase(size_t w, size_t h, pangolin::VideoRecordRepeat& video)
  : w(w), h(h), video(video), video_playback(nullptr),
  //frame("ui.frame", -1, 0, total_frames-1 ),
  finished_(false),
  frame("ui.frame", -1, 0, 0),
  fps_("ui.fps", 0.,0.,0.),
  record_timelapse_frame_skip("viewer.record_timelapse_frame_skip", 1 ),
  end_frame("ui.end_frame", std::numeric_limits<int>::max() ),
  video_wait("ui.wait", true),
  video_newest("ui.newest", false),
  verbose("ui.verbose", true)
{
    // Create OpenGL window - guess sensible dimensions
    int menue_w = 180;
    pangolin::CreateWindowAndBind( "GuiBase", w+menue_w, h);

    // Assume packed OpenGL data unless otherwise specified
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::View& container = pangolin::Display("container");
    container.SetLayout(pangolin::LayoutEqual)
             .SetBounds(0., 1.0, pangolin::Attach::Pix(menue_w), 1.0);

    video_playback = video.Cast<pangolin::VideoPlaybackInterface>();
    const int total_frames = video_playback ?  video_playback->GetTotalFrames() :
      std::numeric_limits<int>::max();

    if( video_playback ) {
        if(total_frames < std::numeric_limits<int>::max() ) {
            std::cout << "Video length: " << total_frames << " frames" << std::endl;
        }
        end_frame = 0;
    }
    end_frame = std::numeric_limits<int>::max();

    const size_t num_streams = video.Streams().size();

    // current frame in memory buffer and displaying.
    pangolin::CreatePanel("ui").SetBounds(0.,1.,0.,pangolin::Attach::Pix(menue_w));

#ifdef CALLEE_HAS_CPP11
    const int FRAME_SKIP = 30;
    const char show_hide_keys[]  = {'1','2','3','4','5','6','7','8','9'};
    const char screenshot_keys[] = {'!','@','#','$','%','^','&','*','('};

    // Show/hide streams
    for(size_t v=0; v < 9; v++) {
        pangolin::RegisterKeyPressCallback(show_hide_keys[v], [&,v](){
            if (v < container.NumChildren()) {
              container[v].ToggleShow();
            }
        } );
        pangolin::RegisterKeyPressCallback(screenshot_keys[v], [&,v](){
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

    buffer.resize(video.SizeBytes()+1);

    // Output details of video stream
    for(size_t s = 0; s < num_streams; ++s) {
      const pangolin::StreamInfo& si = video.Streams()[s];
      std::cout << "Stream " << s << ": " << si.Width() << " x " << si.Height()
        << " " << si.PixFormat().format 
        << " (pitch: " << si.Pitch() << " bytes)" << std::endl;
      if (si.PixFormat().format.compare(
            pangolin::VideoFormatFromString("GRAY16LE").format)==0) {
        iD.push_back(s);
      }
      if (si.PixFormat().format.compare(
            pangolin::VideoFormatFromString("RGB24").format)==0) {
        iRGB.push_back(s);
      }
    }
}

GuiBase::~GuiBase()
{
}

}
