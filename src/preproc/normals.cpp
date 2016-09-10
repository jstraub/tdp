
#include <assert.h>
#include <tdp/preproc/normals.h>

namespace tdp {

#ifdef CUDA_FOUND
void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN) {
  size_t wc = cuD.w_;
  size_t hc = cuD.h_;
  assert(wc%64 == 0);
  assert(hc%64 == 0);
  ManagedDeviceImage<float> cuDu(wc, hc);
  ManagedDeviceImage<float> cuDv(wc, hc);

  Gradient(cuD, cuDu, cuDv);

  float f = cam.params_(0);
  int uc = cam.params_(2);
  int vc = cam.params_(3);
  ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
}
#endif


}
