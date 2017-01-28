#include <tdp/utils/file.h>
#include <sstream>
#include <pangolin/utils/file_utils.h>

namespace tdp {
  std::string MakeUniqueFilename(const std::string& filename)
  {
    if(pangolin::FileExists(filename) ) {
      const size_t dot = filename.find_last_of('.');

      std::string fn;
      std::string ext;

      if(dot == filename.npos) {
        fn = filename;
        ext = "";
      }else{
        fn = filename.substr(0, dot);
        ext = filename.substr(dot);
      }

      int id = 1;
      std::string new_file;
      do {
        id++;
        std::stringstream ss;
        ss << fn << "_" << std::setw(6) << std::setfill('0') << id << ext;
        new_file = ss.str();
      } while(pangolin::FileExists(new_file) );

      return new_file;
    }else{
      return filename;
    }
  }
}
