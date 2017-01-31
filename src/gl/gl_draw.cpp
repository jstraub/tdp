#include <tdp/gl/gl_draw.h>

namespace tdp {
void glDrawPoses(const std::vector<SE3f>& Ts, int step, float scale) {
  for (size_t i=1; i<Ts.size(); ++i) {
    glDrawLine(Ts[i].translation(), Ts[i-1].translation());
  }
  if (step > 0)
    for (size_t i=0; i<Ts.size(); i+=step)
      pangolin::glDrawAxis(Ts[i].matrix(), scale);
}
}
