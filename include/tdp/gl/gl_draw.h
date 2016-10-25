
#pragma once

#include <vector>
#include <pangolin/gl/gldraw.h>
#include <tdp/manifold/SE3.h>

namespace tdp {
void glDrawPoint(GLfloat x1, GLfloat y1) {
  glBegin(GL_POINTS); // render with points
  glVertex2f(x1,y1); //display a point
  glEnd();
}

void glDrawLine(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  pangolin::glDrawLine(a(0),a(1),a(2),b(0),b(1),b(2));
}

void glDrawPoses(const std::vector<SE3f>& Ts, int step=10, float scale=0.1) {
  for (size_t i=1; i<Ts.size(); ++i) {
    glDrawLine(Ts[i].translation(), Ts[i-1].translation());
  }
  if (step > 0)
    for (size_t i=0; i<Ts.size(); i+=step)
      pangolin::glDrawAxis(Ts[i].matrix(), scale);
}

}
