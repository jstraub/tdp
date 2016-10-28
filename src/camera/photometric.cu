#include <tdp/slam/keyframe.h>
#include <tdp/eigen/dense.h>
#include <tdp/cuda/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/camera/camera_base.h>
#include <tdp/camera/camera.h>
#include <tdp/camera/camera_poly.h>
#include <tdp/reductions/reductions.cuh>
#include <tdp/manifold/SE3.h>
#include <tdp/cuda/cuda.cuh>

namespace tdp {

template<int BLK_SIZE, int D, typename Derived>
__global__ void KernelOverlap(
    Image<float> greyA,
    Image<float> greyB,
    Image<Vector3fda> pcA,
    Image<Vector3fda> pcB,
    SE3f T_ab,
    CameraBase<float,D,Derived> camA,
    int N_PER_T,
    Image<Vector3fda> stats,
    float* errB
    ) {
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = pcB.Area();
  const int idE = min(N,(idx+1)*N_PER_T);

  SharedMemory<Vector3fda> smem;
  Vector3fda* sum = smem.getPointer();
  sum[tid] = Vector3fda::Zero();

  for (int id=idS; id<idE; ++id) {
    const int x = id%pcB.w_;
    const int y = id/pcB.w_;
    if (x < pcB.w_ && y < pcB.h_) {
      Vector3fda pB = pcB(x,y);
      if (IsValidData(pB)) {
        Vector3fda pBinA = T_ab*pB;
        Eigen::Vector2f uv = camA.Project(pBinA);
        if (greyA.Inside(uv)) {
          Vector3fda pA = pcA(floor(uv(0)), floor(uv(1)));
          if ((pBinA-pA).norm() < 0.03) {
            //          if (tid % 10 == 0)
            //          printf("%f %f %d %d %d\n", uv(0), uv(1), x, y, id);
            float diff = greyA.GetBilinear(uv)-greyB(x,y);
            //          float diff = greyB(x,y);
            float rmse = diff*diff;
            if (errB) errB[id] = sqrt(rmse);
            sum[tid](0) += rmse;
            sum[tid](1) += 1;
          }
        }
        sum[tid](2) += 1;
      }
    }
  }
  SumPyramidReduce<Vector3fda, BLK_SIZE>(tid, sum, stats.ptr_);
}

template <int D, class Derived>
void OverlapGpu(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA, 
    const Image<Vector3fda>& pcB, 
    const SE3f& T_ab, 
    const CameraBase<float,D,Derived>& camA, float& overlap, float& rmse, 
    Image<float>* errB) {
  
  const size_t BLK_SIZE = 32;
  size_t N = pcB.Area();
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,BLK_SIZE);
  ManagedDeviceImage<Vector3fda> out(1,1);
  cudaMemset(out.ptr_, 0, out.SizeBytes());

  KernelOverlap<BLK_SIZE,D,Derived><<<blocks,threads,
    BLK_SIZE*sizeof(Vector3fda)>>>(greyA, greyB, pcA, pcB, T_ab, camA, 10,
        out, errB ? errB->ptr_ : nullptr);
  checkCudaErrors(cudaDeviceSynchronize());

  ManagedHostImage<Vector3fda> stats(1,1);
  stats.CopyFrom(out, cudaMemcpyDeviceToHost);

  overlap = stats[0](1) / stats[0](2); 
  rmse = sqrtf(stats[0](0) / stats[0](2)); 


}

template 
void OverlapGpu(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA, 
    const Image<Vector3fda>& pcB, 
    const SE3f& T_ab, 
    const BaseCameraf& camA, float& overlap, float& rmse, 
    Image<float>* errB);
template 
void OverlapGpu(const Image<float>& greyA, const Image<float>& greyB,
    const Image<Vector3fda>& pcA, 
    const Image<Vector3fda>& pcB, 
    const SE3f& T_ab, 
    const BaseCameraPoly3f& camA, float& overlap, float& rmse, 
    Image<float>* errB);


}
