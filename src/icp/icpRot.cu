/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <assert.h>
#include <tdp/eigen/dense.h>
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/reductions/reductions.cuh>
#include <tdp/manifold/SE3.h>
#include <tdp/cuda/cuda.cuh>

namespace tdp {

// T_mc: T_model_current
template<int BLK_SIZE, int D, typename Derived>
__global__ void KernelICPStepRotation(
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    SE3f T_mo, 
    SE3f T_cm,
    const CameraBase<float,D,Derived> cam,
    float dotThr,
    int N_PER_T,
    Image<float> out
    ) {
  assert(BLK_SIZE >=10);
  const int tid = threadIdx.x;
  const int id_ = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = id_*N_PER_T;
  const int idE = min((int)pc_o.Area(),(id_+1)*N_PER_T);
  SharedMemory<Vector10fda> smem;
  Vector10fda* sum = smem.getPointer();
  sum[tid] = Vector10fda::Zero();
  for (int id=idS; id<idE; ++id) {
    const int idx = id%pc_o.w_;
    const int idy = id/pc_o.w_;
    // project current point into model frame to get association
    if (idx >= pc_o.w_ || idy >= pc_o.h_) continue;
    Vector3fda pc_oi = pc_o(idx,idy);
    Vector3fda pc_o_in_m = T_mo * pc_oi ;
    // project into model camera
    // TODO: doing the association the other way around might be more
    // stable since the model depth is smoothed
    Vector2fda x_o_in_m = cam.Project(T_cm * pc_o_in_m);
    const int u = floor(x_o_in_m(0)+0.5f);
    const int v = floor(x_o_in_m(1)+0.5f);
    if (0 <= u && u < pc_o.w_ && 0 <= v && v < pc_o.h_
        && pc_oi(2) > 0. && pc_o_in_m(2) > 0.
        && IsValidData(pc_o_in_m)) {
      // found association -> check thresholds;
      Vector3fda n_oi = n_o(idx,idy);
      Vector3fda n_o_in_m = T_mo.rotation() * n_oi;
//      Vector3fda n_o_in_m = n_o(idx,idy);
      Vector3fda n_mi = n_m(u,v);
      const float dot  = n_mi.dot(n_o_in_m);
      if (dot > dotThr && IsValidData(n_mi)) {
        // association is good -> accumulate
        // TODO: test: association uses T_mo but computation of N does
        // not  since we can get R in closed form from this.
        sum[tid](0) += n_mi(0)*n_oi(0);
        sum[tid](1) += n_mi(0)*n_oi(1);
        sum[tid](2) += n_mi(0)*n_oi(2);
        sum[tid](3) += n_mi(1)*n_oi(0);
        sum[tid](4) += n_mi(1)*n_oi(1);
        sum[tid](5) += n_mi(1)*n_oi(2);
        sum[tid](6) += n_mi(2)*n_oi(0);
        sum[tid](7) += n_mi(2)*n_oi(1);
        sum[tid](8) += n_mi(2)*n_oi(2);
        sum[tid](9) += 1.; // to get number of data points
      }
    }
  }
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      sum[tid] += sum[tid+s];
    }
    __syncthreads();
  }
  if(tid < 10) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd(&out[tid], sum[0](tid)+sum[1](tid));
  }
}

template<int D, typename Derived>
void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    const SE3f& T_mo, 
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count
    ) {
  const size_t BLK_SIZE = 32;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,pc_o.Area()/10,BLK_SIZE);
  ManagedDeviceImage<float> out(10,1);
  cudaMemset(out.ptr_, 0, 10*sizeof(float));

  KernelICPStepRotation<BLK_SIZE,D,Derived><<<blocks,threads,
    BLK_SIZE*sizeof(Vector10fda)>>>(
        n_m,n_o,pc_o,T_mo, T_cm, cam,
        dotThr,10,out);
  checkCudaErrors(cudaDeviceSynchronize());
  ManagedHostImage<float> nUpperTri(10,1);
  cudaMemcpy(nUpperTri.ptr_,out.ptr_,10*sizeof(float), cudaMemcpyDeviceToHost);

  //for (int i=0; i<29; ++i) std::cout << sumAb[i] << "\t";
  //std::cout << std::endl;
  N.fill(0.);
  int k = 0;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      N(i,j) = nUpperTri[k++];
    }
  }
  count = nUpperTri[9];
  //std::cout << ATA << std::endl << ATb.transpose() << std::endl;
  //std::cout << "\terror&count " << error << " " << count << std::endl;
}

template void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    const SE3f& T_mo, 
    const SE3f& T_cm,
    const BaseCameraf& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count);
template void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_o,
    Image<Vector3fda> pc_o,
    const SE3f& T_mo, 
    const SE3f& T_cm,
    const BaseCameraPoly3f& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count);

}


