
#pragma once

#include <vector>
#include <pangolin/gl/gldraw.h>
#include <tdp/eigen/dense.h>
#include <tdp/manifold/SE3.h>

namespace tdp {

inline void glDrawPoint(GLfloat x1, GLfloat y1) {
  glBegin(GL_POINTS); // render with points
  glVertex2f(x1,y1); //display a point
  glEnd();
}

inline void glDrawLine(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  pangolin::glDrawLine(a(0),a(1),a(2),b(0),b(1),b(2));
}

inline void glDrawLine(const Vector3fda& p1, const Vector3fda& p2 )
{
    pangolin::glDrawLine((GLfloat)p1(0), (GLfloat)p1(1), (GLfloat)p1(2),
                         (GLfloat)p2(0), (GLfloat)p2(1), (GLfloat)p2(2));
}

void glDrawPoses(const std::vector<SE3f>& Ts, int step=10, float
    scale=0.1);

}
