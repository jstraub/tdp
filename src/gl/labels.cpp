#include <tdp/config.h>
#include <tdp/gl/labels.h>
#include <pangolin/utils/file_utils.h>

namespace tdp {

Labels* Labels::labels_ = nullptr;

Labels* Labels::Instance() {
  if (!labels_)
    labels_ = new Labels;
  return labels_;
}

Labels::Labels() {
  std::string file = SHADER_DIR+std::string("/labels/randi.png");
  std::cout << "loading labels texture" << std::endl;
  std::cout << file << std::endl;
  labelsImg_ = pangolin::LoadImage(file);
}

}
