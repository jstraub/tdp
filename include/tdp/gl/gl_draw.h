
#pragma once

#include <pangolin/gl/gldraw.h>

namespace tdp {
void glDrawPoint(GLfloat x1, GLfloat y1) {
  glBegin(GL_POINTS); // render with points
  glVertex2f(x1,y1); //display a point
  glEnd();
}

}
