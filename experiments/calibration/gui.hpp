

class GUI {
 public:
  GUI();
  ~GUI();

  pangolin::View& container() {
    return pangolin::Display("container")
  }
 private:
  size_t w;
  size_t h;
  pangolin::VideoRecordRepeat& video;
  // Setup resizable views for video streams
  std::vector<pangolin::GlPixFormat> glfmt;
  std::vector<std::pair<float,float> > gloffsetscale;
  std::vector<size_t> strides;
  std::vector<pangolin::ImageViewHandler> handlers;
  std::vector<tdp::QuickView*> streamViews;

  pangolin::Var<int> frame;
  pangolin::Var<int>  record_timelapse_frame_skip;
  pangolin::Var<int>  end_frame;
  pangolin::Var<bool> video_wait;
  pangolin::Var<bool> video_newest;

};

