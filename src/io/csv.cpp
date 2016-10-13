#include <tdp/io/csv.h>
#include <iostream>
#include <fstream>

namespace tdp {

void LoadPointCloudCsv(
    const std::string& path,
    ManagedHostImage<Vector3fda>& verts) {
  std::ifstream in(path);
  size_t w=0;
  size_t h=0;
  in >> w >> h;
  verts.Reinitialise(w,h);
  for (size_t i=0; i<verts.Area(); ++i) {
    in >> verts[i](0) >> verts[i](1) >> verts[i](2);
  }
  in.close();
}

void SavePointCloudCsv(
    const std::string& path,
    const Image<Vector3fda>& verts,
    std::vector<std::string>& comments) {
  std::ofstream out(path);
  out << verts.w_ << " " << verts.h_ << std::endl;
  for (size_t i=0; i<verts.Area(); ++i) {
    out << verts[i](0) << " "  
      << verts[i](1) << " "
      << verts[i](2) << std::endl;
  }
  out.close();
}

}
