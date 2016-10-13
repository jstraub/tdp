#include <tdp/config.h>
#include <tdp/gl/matcap.h>
#include <pangolin/utils/file_utils.h>

namespace tdp {

Matcap* Matcap::matcap_ = nullptr;

Matcap* Matcap::Instance() {
  if (!matcap_)
    matcap_ = new Matcap;
  return matcap_;
}

Matcap::Matcap() {
  std::string matcapRoot = SHADER_DIR+std::string("/matcap/");

  std::vector<std::string> files;
  pangolin::FilesMatchingWildcard(matcapRoot+"*.png",files);

  std::cout << "loading matcap textures" << std::endl;
  for (auto& file : files) {
    std::cout << file << std::endl;
    matcapImgs_.push_back(pangolin::LoadImage(file));
  }
}

}
