
#pragma once
#include <vector>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>
#include <pangolin/var/var.h>
#include <pangolin/display/display.h>
#include <pangolin/video/video.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/utils/timer.h>

#include <tdp/gui/quickView.h>
#include <tdp/eigen/dense.h>

namespace tdp {

class GuiBase {
 public:
  GuiBase(size_t w, size_t h, pangolin::VideoRecordRepeat& video);
  ~GuiBase();

  void NextFrames() {
    if(frame.GuiChanged()) {
      if(video_playback) {
        frame = video_playback->Seek(frame) -1;
      }
      end_frame = frame + 1;
    }
    if ( frame < end_frame ) {
      GrabFrames();
    }
    t_prev_ = t_;
    t_ = pangolin::Time_us(pangolin::TimeNow());
    fps_ = 1./((t_-t_prev_)*1e-6);
  }

  void SeekFrames(size_t destFrame) {
    if (video_playback && destFrame < end_frame ) {
      video_playback->Seek(frame);
      GrabFrames();
    }
//    end_frame = frame + 1;
  }

  pangolin::Image<uint8_t>& Image(int i) { return images[i]; } 

  bool ImageD(tdp::Image<uint16_t>& d, size_t camId=0, 
      int64_t* t_host_us=nullptr) const {
    if (camId >= iD.size()) return false;
    d = tdp::Image<uint16_t>(images[iD[camId]].w, images[iD[camId]].h,
        images[iD[camId]].pitch, 
        reinterpret_cast<uint16_t*>(images[iD[camId]].ptr));
    if (t_host_us) *t_host_us = t_host_us_[iD[camId]];
    return true; 
  } 
  bool ImageRGB(tdp::Image<tdp::Vector3bda>& rgb, size_t camId=0, 
      int64_t* t_host_us=nullptr) const {
    if (camId >= iRGB.size()) return false;
    rgb = tdp::Image<tdp::Vector3bda>(images[iRGB[camId]].w, 
        images[iRGB[camId]].h,
        images[iRGB[camId]].pitch, 
        reinterpret_cast<tdp::Vector3bda*>(images[iRGB[camId]].ptr));
    if (t_host_us) *t_host_us = t_host_us_[iRGB[camId]];
    return true; 
  } 

  pangolin::View& container() {
    return pangolin::Display("container");
  }

  bool pause() { 
    end_frame = (frame < end_frame) ? frame : std::numeric_limits<int>::max(); 
  }
  bool paused() { return frame == end_frame; }
  bool finished() { return finished_; }

  std::vector<int> iRGB, iD;
  std::vector<int64_t> t_host_us_;

  bool finished_;
  pangolin::Var<int> frame;
  pangolin::Var<float> fps_;
  pangolin::Var<int>  record_timelapse_frame_skip;
  pangolin::Var<int>  end_frame;
  pangolin::Var<bool> video_wait;
  pangolin::Var<bool> video_newest;
  pangolin::Var<bool> verbose;

  pangolin::VideoRecordRepeat& video;
  std::vector<pangolin::Image<unsigned char> > images;

 protected:
  size_t w;
  size_t h;
  pangolin::VideoPlaybackInterface* video_playback;

  std::vector<unsigned char> buffer;

  int64_t t_prev_;
  int64_t t_;

  void GrabFrames() {
    if( video.Grab(&buffer[0], images, video_wait, video_newest) ) {
      frame = frame +1;

      pangolin::json::value props = pangolin::GetVideoFrameProperties(&video);
      //        std::cout << props.serialize(true) << std::endl;
      t_host_us_.clear();
      if (props.contains("streams")) {
        for (size_t i=0; i<images.size(); ++i) {
          if (props["streams"][i].contains("hosttime_us")) {
            t_host_us_.push_back(props["streams"][i]["hosttime_us"].get<int64_t>());
          }
        }
      } else {
        std::cout << "could not find strem properties in: " << std::endl
          << props.serialize(true) << std::endl;
        for (size_t i=0; i<images.size(); ++i) {
          t_host_us_.push_back(0);
        }
      }
    } else {
      finished_ = true;
    }
  }
};

}
