
#include <tdp/eigen/dense.h>
#include <tdp/cuda.h>
#include <tdp/nvidia/helper_cuda.h>
#include <tdp/image.h>
#include <tdp/managed_image.h>
#include <tdp/camera.h>
#include <tdp/reductions.cuh>

namespace tdp {

// R_mc: R_model_current
template<int BLK_SIZE>
__global__ void KernelICPStep(
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_c,
    Image<Vector3fda> n_c,
    Matrix3fda R_mc, 
    Vector3fda t_mc, 
    const Camera<float> cam,
    float dotThr,
    float distThr,
    int N_PER_T,
    Image<float> out
    ) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = tid*N_PER_T;
  const int N = pc_m.w_*pc_m.h_;
  const int idE = min(N,(tid+1)*N_PER_T);
  __shared__ Eigen::Matrix<float,22,1,Eigen::DontAlign> sum[BLK_SIZE];

  sum[threadIdx.x] = Eigen::Matrix<float,22,1,Eigen::DontAlign>::Zero();

  for (int id=idS; id<idE; ++id) {
    const int idx = id%pc_c.w_;
    const int idy = id/pc_c.w_;
    // project current point into model frame to get association
    if (idx < pc_c.w_ && idy < pc_c.h_) {
      Vector3fda pc_c_in_m = R_mc * pc_c(idx,idy) + t_mc;
      // project into model camera
      Vector2fda x_c_in_m = cam.Project(pc_c_in_m);
      int u = floor(x_c_in_m(0)+0.5f);
      int v = floor(x_c_in_m(1)+0.5f);
      if (0 <= u && u < pc_m.w_ && 0 <= v && v < pc_m.h_ && !isnan(pc_c_in_m(0))) {
        // found association -> check thresholds;
        Vector3fda n_c_in_m = R_mc * n_c(idx,idy);
        Vector3fda n_mi = n_m(idx,idy);
        Vector3fda pc_mi = pc_m(idx,idy);
        float dot  = n_mi.dot(n_c_in_m);
        if (dot > dotThr && !isnan(pc_mi(0))) {
          // association is good -> accumulate
          // if we found a valid association accumulate the A and b for A x = b
          // where x \in se{3} as well as the residual error
          float ab[7];      
          Eigen::Map<Vector3fda> top(&(ab[0]));
          Eigen::Map<Vector3fda> bottom(&(ab[3]));
          top = (R_mc * pc_c(idx,idy)).cross(n_mi);
          bottom = n_mi;
          ab[6] = n_mi.dot(pc_mi-pc_c_in_m);
          Eigen::Matrix<float,22,1,Eigen::DontAlign> upperTriangle;
          int k=0;
#pragma unroll
          for (int i=0; i<7; ++i) {
            for (int j=i; j<7; ++j) {
              upperTriangle(k++) = ab[i]*ab[j];
            }
          }
          upperTriangle(22) = 1.; // to get number of data points
          sum[threadIdx.x] += upperTriangle;
        }
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
  if(threadIdx.x < 22) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<float>(&out[threadIdx.x], sum[0](threadIdx.x)+sum[1](threadIdx.x));
  }
}

void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_c,
    Image<Vector3fda> n_c,
    Matrix3fda& R_mc, 
    Vector3fda& t_mc, 
    const Camera<float>& cam,
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& count
    ) {
  size_t N = pc_m.w_*pc_m.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  ManagedDeviceImage<float> out(22,1);
  cudaMemset(out.ptr_, 0, 22*sizeof(float));

  KernelICPStep<256><<<blocks,threads>>>(pc_m,n_m,pc_c,n_c,R_mc,t_mc,cam,dotThr,distThr,10,out);
  checkCudaErrors(cudaDeviceSynchronize());
  Eigen::Matrix<float,22,1,Eigen::DontAlign> sumAb;
  cudaMemcpy(&sumAb(0),out.ptr_,22*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << sumAb.transpose() << std::endl;
  int prevRowStart = 0;
  for (int i=0; i<6; ++i) {
    ATb(i) = sumAb(prevRowStart+7-i-1);
    for (int j=i; j<6; ++j) {
      ATA(i,j) = sumAb(prevRowStart+j-i);
    }
    prevRowStart += 7-i;
  }
  ATA += ATA.transpose();
  count = sumAb(21);
}

}
