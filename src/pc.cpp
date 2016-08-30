#include <assert>
#include <tdp/depth.h>
#include <tdp/image.h>

namespace tdp {

void PyramidDepth2PCs(
    const Pyramid<float>& d,
    const Camera<float>& cam,
    Pyramid<Vector3fda>& pc
    ) {
  assert(d.Lvls() == pc.Lvls());
  for (size_t lvl=0; lvl<d.Lvls(); ++lvl) {
    Image<Vector3fda> pc_i = pc.GetImage(lvl);
    Image<Vector3fda> d_i = d.GetImage(lvl);
    Depth2PC(d_i, cam, pc_i);
  }
}

}
