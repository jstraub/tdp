
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

void glDrawPoses(const std::vector<SE3f>& Ts, int step=10, float scale=0.1) {
  for (size_t i=1; i<Ts.size(); ++i) {
    pangolin::glDrawLine(
        Ts[i].translation()(0),
        Ts[i].translation()(1),
        Ts[i].translation()(2),
        Ts[i-1].translation()(0),
        Ts[i-1].translation()(1),
        Ts[i-1].translation()(2));
  }
  for (size_t i=0; i<Ts.size(); i+=step)
    pangolin::glDrawAxis(Ts[i].matrix(), scale);
}

}
