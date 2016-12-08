
#include <vector>
#include <Eigen/Dense>
#include <tdp/eigen/dense.h>
#include <tdp/distributions/vmf.h>
#include <tdp/manifold/SO3.h>
#include <tdp/data/managed_image.h>
#include <tdp/distributions/vmf_mm.h>

namespace tdp {

void MAPLabelAssignvMFMM( 
    std::vector<vMF<float,3>>& vmfs,
    const SO3fda& R_nvmf,
    const Image<Vector3fda>& cuN,
    Image<uint16_t>& cuZ) {

  ManagedHostImage<Vector3fda> tauMu(vmfs.size());
  ManagedHostImage<float> logPi(vmfs.size());
  ManagedDeviceImage<Vector3fda> cuTauMu(vmfs.size());
  ManagedDeviceImage<float> cuLogPi(vmfs.size());

  for (size_t i=0; i<vmfs.size(); ++i) {
    logPi[i] = log(vmfs[i].GetPi());
    tauMu[i] = vmfs[i].GetTau()*(R_nvmf*vmfs[i].GetMu());
  }

  cuLogPi.CopyFrom(logPi);
  cuTauMu.CopyFrom(tauMu);

  MAPLabelAssignvMFMM(cuN, cuTauMu, cuLogPi, cuZ);
}


}
