
#pragma once
#include <vector>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>
#include <pangolin/var/var.h>
#include <pangolin/display/display.h>
#include <pangolin/video/video.h>
#include <pangolin/video/video_record_repeat.h>

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
      if( video.Grab(&buffer[0], images, video_wait, video_newest) ) {
        frame = frame +1;
      }
    }
  }

  pangolin::Image<uint8_t>& Image(int i) { return images[i]; } 

  bool ImageD(tdp::Image<uint16_t>& d, size_t camId=0) const {
    if (camId >= iD.size()) return false;
    d = tdp::Image<uint16_t>(images[iD[camId]].w, images[iD[camId]].h,
        images[iD[camId]].pitch, reinterpret_cast<uint16_t*>(images[iD[camId]].ptr));
    return true; 
  } 
  bool ImageRGB(tdp::Image<tdp::Vector3bda>& rgb, size_t camId=0) const {
    if (camId >= iRGB.size()) return false;
    rgb = tdp::Image<tdp::Vector3bda>(images[iRGB[camId]].w, images[iRGB[camId]].h,
        images[iRGB[camId]].pitch, 
        reinterpret_cast<tdp::Vector3bda*>(images[iRGB[camId]].ptr));
    return true; 
  } 

  pangolin::View& container() {
    return pangolin::Display("container");
  }

  std::vector<int> iRGB, iD;

  pangolin::Var<int> frame;
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
};

}
