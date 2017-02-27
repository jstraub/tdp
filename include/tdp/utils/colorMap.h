/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <tdp/eigen/dense.h>
namespace tdp {

Vector3bda ColorMapHot(float cVal) {
    return Vector3bda (
          cVal<0.20 ? 255*cVal*5 : 255,
          cVal<0.40 ? 0 : cVal < 0.80 ? 255*(cVal-.4)*2.5 : 255,
          cVal<0.80 ? 0 : 255*(cVal-0.8)*5 );
}

void glColorHot(float cVal) {
  glColor3f(cVal<0.20 ? 255*cVal*5 : 255,
            cVal<0.40 ? 0 : cVal < 0.80 ? 255*(cVal-.4)*2.5 : 255,
            cVal<0.80 ? 0 : 255*(cVal-0.8)*5 );
}

void glColorHot(float cVal, float alpha) {
  glColor4f(cVal<0.20 ? 255*cVal*5 : 255,
            cVal<0.40 ? 0 : cVal < 0.80 ? 255*(cVal-.4)*2.5 : 255,
            cVal<0.80 ? 0 : 255*(cVal-0.8)*5 , alpha);
}

}
