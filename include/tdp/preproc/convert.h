/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <tdp/data/image.h>
#include <tdp/data/pyramid.h>
#include <tdp/data/volume.h>

namespace tdp {

template<typename Tin, typename Tout>
void Convert(Tin* in, Tout* out, float scale, float offset, size_t N);

template<typename Tin, typename Tout>
inline void Convert(const Image<Tin>& in, const Image<Tout>& out, 
    float scale = 1., float offset = 0.) {
  Convert<Tin,Tout>(in.ptr_, out.ptr_, scale, offset,
      std::min(out.Area(),in.Area())); 
}

template<typename Tin, typename Tout, int LEVELS>
inline void Convert(const Pyramid<Tin,LEVELS>& in, 
    const Pyramid<Tout,LEVELS>& out, 
    float scale = 1., float offset = 0.) {
  Convert<Tin,Tout>(in.ptr_, out.ptr_, scale, offset, 
      std::min(out.NumElems(),in.NumElems())); 
}

template<typename Tin, typename Tout>
inline void Convert(const Volume<Tin>& in, const Volume<Tout>& out, 
    float scale = 1., float offset = 0.) {
  Convert<Tin,Tout>(in.ptr_, out.ptr_, scale, offset, 
      std::min(out.Volume(),in.Volume())); 
}

}
