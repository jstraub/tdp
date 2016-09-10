
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

class GUI {
 public:
  GUI(size_t w, size_t h, pangolin::VideoRecordRepeat& video);
  ~GUI();

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

  bool ImageD(tdp::Image<uint16_t>& d) {
    if (iD < 0) return false;
    d = tdp::Image<uint16_t>(images[iD].w, images[iD].h,
        images[iD].pitch, reinterpret_cast<uint16_t*>(images[iD].ptr));
    return true; 
  } 
  bool ImageRGB(tdp::Image<tdp::Vector3bda>& rgb) {
    if (iRGB < 0) return false;
    rgb = tdp::Image<tdp::Vector3bda>(images[iRGB].w, images[iRGB].h,
        images[iRGB].pitch, 
        reinterpret_cast<tdp::Vector3bda*>(images[iRGB].ptr));
    return true; 
  } 

  void ShowFrames() {
    for(unsigned int i=0; i<images.size(); ++i)
    {
      if(container()[i].IsShown()) {
        pangolin::Image<unsigned char>& image = images[i];
        streamViews[i]->SetImage(image, glfmt[i], strides[i]);
      }
    }
  }

  pangolin::View& container() {
    return pangolin::Display("container");
  }

  int iRGB, iD;

  pangolin::Var<int> frame;
  pangolin::Var<int>  record_timelapse_frame_skip;
  pangolin::Var<int>  end_frame;
  pangolin::Var<bool> video_wait;
  pangolin::Var<bool> video_newest;
  pangolin::Var<bool> verbose;

  pangolin::Var<bool> useMatCap;

 private:
  size_t w;
  size_t h;
  pangolin::VideoRecordRepeat& video;
  pangolin::VideoPlaybackInterface* video_playback;
  // Setup resizable views for video streams
  std::vector<pangolin::GlPixFormat> glfmt;
  std::vector<std::pair<float,float> > gloffsetscale;
  std::vector<size_t> strides;
  std::vector<pangolin::ImageViewHandler> handlers;
  std::vector<tdp::QuickView*> streamViews;

  std::vector<unsigned char> buffer;
  std::vector<pangolin::Image<unsigned char> > images;



};

