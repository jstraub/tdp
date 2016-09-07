
#include <tdp/eigen/dense.h>
#include <tdp/image.h>
#include <tdp/managed_image.h>
#include <tdp/cuda.h>
#include <tdp/reductions.cuh>

#include <tdp/sufficientStats.h>

namespace tdp {

template<typename T, size_t D, int BLK_SIZE>
__global__
void KernelSufficientStats1stOrder(
    Image<Eigen::Matrix<T,D,1,Eigen::DontAlign>> I,
    Image<Eigen::Matrix<T,D+1,1,Eigen::DontAlign>> Isum,
    Image<uint16_t> z,
    uint16_t k,
    int N_PER_T
    ) {
  __shared__ Eigen::Matrix<T,D+1,1,Eigen::DontAlign> sum[BLK_SIZE];
  //const int tid = threadIdx.x;
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = I.Area();
  const int idE = min(N,(idx+1)*N_PER_T);
  // caching 
  //if (tid==0) printf("%d <? %d %d\n",idS,N,N_PER_T);
  sum[tid] = Eigen::Matrix<T,D+1,1,Eigen::DontAlign>::Zero();
  for(int i=idS; i<idE; ++i) {
    if (!isNan(I[i]) && (!z.ptr_ || (z.ptr_ && z[i]==k))) {
      Eigen::Matrix<T,D+1,1,Eigen::DontAlign> xi;
      xi.topRows(D) = I[i];
      xi(D) = 1.;
      sum[tid] += xi;
    }
  }
  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      sum[tid] += sum[tid+s];
    }
    __syncthreads();
  }
  if(tid < D+1) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(&(Isum.ptr_[0](tid)), sum[0](tid)+sum[1](tid));
  }
}


Vector4fda SufficientStats1stOrder(const Image<Vector3fda>& I) {
  size_t N = I.Area();
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<Vector4fda> Isum(1,1);
  cudaMemset(Isum.ptr_, 0, sizeof(Vector4fda));
  Image<uint16_t> z;
  KernelSufficientStats1stOrder<float,3,256><<<blocks,threads>>>(I,Isum,z,0,N/10);
  checkCudaErrors(cudaDeviceSynchronize());
  Vector4fda sum = Vector4fda::Zero();
  cudaMemcpy(&sum,Isum.ptr_,sizeof(Vector4fda), cudaMemcpyDeviceToHost);
  return sum;
}

Eigen::Matrix<float,4,Eigen::Dynamic, Eigen::DontAlign> SufficientStats1stOrder(
    const Image<Vector3fda>& I, const Image<uint16_t> z, uint16_t K) {

  size_t N = I.Area();
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<Vector4fda> Iss(K,1);
  cudaMemset(Iss.ptr_, 0, K*sizeof(Vector4fda));

  for (uint16_t k=0; k<K; ++k) {
    Image<Vector4fda> Issk(1,1,&Iss[k]);
    KernelSufficientStats1stOrder<float,3,256><<<blocks,threads>>>(I,Issk,z,k,N/10);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  Eigen::Matrix<float,4,Eigen::Dynamic, Eigen::DontAlign> ss(4,K);
  cudaMemcpy(ss.data(),Iss.ptr_, K*sizeof(Vector4fda), cudaMemcpyDeviceToHost);
  return ss;
}

}
