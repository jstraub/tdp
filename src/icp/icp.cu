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
#include <tdp/reductions/reductions.cuh>
#include <tdp/manifold/SE3.h>

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
  assert(BLK_SIZE >=29);
  const int tid = threadIdx.x;
  const int id_ = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = id_*N_PER_T;
  const int N = pc_m.w_*pc_m.h_;
  const int idE = min(N,(id_+1)*N_PER_T);
  __shared__ Eigen::Matrix<float,29,1,Eigen::DontAlign> sum[BLK_SIZE];
  sum[tid] = Eigen::Matrix<float,29,1,Eigen::DontAlign>::Zero();
  for (int id=idS; id<idE; ++id) {
    const int idx = id%pc_c.w_;
    const int idy = id/pc_c.w_;
    // project current point into model frame to get association
    if (idx < pc_c.w_ && idy < pc_c.h_ ) {
      Vector3fda pc_ci = pc_c(idx,idy);
      if (!IsValidData(pc_ci)) continue;
      Vector3fda pc_c_in_m = R_mc * pc_ci + t_mc;
      // project into model camera
      Vector2fda x_c_in_m = cam.Project(pc_c_in_m);
      const int u = floor(x_c_in_m(0)+0.5f);
      const int v = floor(x_c_in_m(1)+0.5f);
      if (0 <= u && u < pc_m.w_ && 0 <= v && v < pc_m.h_
          && pc_ci(2) > 0. && pc_c_in_m(2) > 0.
          && IsValidData(pc_c_in_m)) {
        // found association -> check thresholds;
        Vector3fda n_c_in_m = R_mc * n_c(idx,idy);
        Vector3fda n_mi = n_m(u,v);
        Vector3fda pc_mi = pc_m(u,v);
        const float dot  = n_mi.dot(n_c_in_m);
        const float dist = (pc_mi-pc_c_in_m).norm();
        if (dot > dotThr && dist < distThr && IsValidData(pc_mi)) {
          // association is good -> accumulate
          // if we found a valid association accumulate the A and b for A x = b
          // where x \in se{3} as well as the residual error
          float ab[7];      
          Eigen::Map<Vector3fda> top(&(ab[0]));
          Eigen::Map<Vector3fda> bottom(&(ab[3]));
          // as in mp3guy: 
          top = (pc_c_in_m).cross(n_mi);
          bottom = n_mi;
          ab[6] = n_mi.dot(pc_mi-pc_c_in_m);
          Eigen::Matrix<float,29,1,Eigen::DontAlign> upperTriangle;
          int k=0;
#pragma unroll
          for (int i=0; i<7; ++i) {
            for (int j=i; j<7; ++j) {
              upperTriangle(k++) = ab[i]*ab[j];
            }
          }
          upperTriangle(28) = 1.; // to get number of data points
          sum[tid] += upperTriangle;
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
  if(tid < 29) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd(&out[tid], sum[0](tid)+sum[1](tid));
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
    float& error,
    float& count
    ) {
  size_t N = pc_m.w_*pc_m.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,32);
  ManagedDeviceImage<float> out(29,1);
  cudaMemset(out.ptr_, 0, 29*sizeof(float));

  KernelICPStep<32><<<blocks,threads>>>(pc_m,n_m,pc_c,n_c,R_mc,t_mc,cam,
      dotThr,distThr,10,out);
  checkCudaErrors(cudaDeviceSynchronize());
  ManagedHostImage<float> sumAb(29,1);
  cudaMemcpy(sumAb.ptr_,out.ptr_,29*sizeof(float), cudaMemcpyDeviceToHost);

  //for (int i=0; i<29; ++i) std::cout << sumAb[i] << "\t";
  //std::cout << std::endl;
  ATA.fill(0.);
  ATb.fill(0.);
  int k = 0;
  for (int i=0; i<6; ++i) {
    for (int j=i; j<7; ++j) {
      float val = sumAb[k++];
      if (j==6)  {
        ATb(i) = val;
      } else {
        ATA(i,j) = val;
        ATA(j,i) = val;
      }
    }
  }
  count = sumAb[28];
  error = sumAb[27]/count;
  //std::cout << ATA << std::endl << ATb.transpose() << std::endl;
  //std::cout << "\terror&count " << error << " " << count << std::endl;
}

// R_mc: R_model_current
template<int BLK_SIZE>
__global__ void KernelICPVisualizeAssoc(
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
    Image<float> assoc_m,
    Image<float> assoc_c
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = pc_m.w_*pc_m.h_;
  const int idE = min(N,(idx+1)*N_PER_T);

  for (int id=idS; id<idE; ++id) {
    const int idx = id%pc_c.w_;
    const int idy = id/pc_c.w_;
    Vector3fda pc_ci = pc_c(idx,idy);
    // project current point into model frame to get association
    if (idx < pc_c.w_ && idy < pc_c.h_ && IsValidData(pc_ci)) {
      Vector3fda pc_c_in_m = R_mc * pc_ci + t_mc;
      // project into model camera
      Vector2fda x_c_in_m = cam.Project(pc_c_in_m);
      int u = floor(x_c_in_m(0)+0.5f);
      int v = floor(x_c_in_m(1)+0.5f);
      if (0 <= u && u < pc_m.w_ && 0 <= v && v < pc_m.h_
          && pc_ci(2) > 0. && pc_c_in_m(2) > 0.
          && IsValidData(pc_c_in_m)) {
        // found association -> check thresholds;
        Vector3fda n_c_in_m = R_mc * n_c(idx,idy);
        Vector3fda n_mi = n_m(u,v);
        Vector3fda pc_mi = pc_m(u,v);
        float dot  = n_mi.dot(n_c_in_m);
        float dist = (pc_mi-pc_c_in_m).norm();
        if (dot > dotThr && dist < distThr && IsValidData(pc_mi)) {
          // association is good -> accumulate
          //assoc_m(u,v) = n_mi.dot(-pc_mi+pc_c_in_m);
          //assoc_c(idx,idy) = n_mi.dot(-pc_mi+pc_c_in_m);
          assoc_m(u,v) = (-pc_mi+pc_c_in_m).norm();
          assoc_c(idx,idy) = (-pc_mi+pc_c_in_m).norm();
//          if (threadIdx.x < 3) printf("%d,%d and %d,%d\n", idx,idy,u,v);
        }
      }
    }
  }
}

void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_c,
    Image<Vector3fda> n_c,
    SE3f& T_mc,
    const Camera<float>& cam,
    float angleThr,
    float distThr,
    Image<float>& assoc_m,
    Image<float>& assoc_c
    ) {
  size_t N = pc_m.w_*pc_m.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  Matrix3fda R_mc = T_mc.rotation().matrix();
  Vector3fda t_mc = T_mc.translation();
  KernelICPVisualizeAssoc<256><<<blocks,threads>>>(pc_m,n_m,pc_c,n_c,
      R_mc,t_mc,cam,
      cos(angleThr*M_PI/180.),distThr,10,assoc_m, assoc_c);
  checkCudaErrors(cudaDeviceSynchronize());
}

// R_mc: R_model_current
template<int BLK_SIZE>
__global__ void KernelICPStepRotation(
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_c,
    Image<Vector3fda> pc_c,
    Matrix3fda R_mc, 
    Vector3fda t_mc, 
    const Camera<float> cam,
    float dotThr,
    int N_PER_T,
    Image<float> out
    ) {
  assert(BLK_SIZE >=7);
  const int tid = threadIdx.x;
  const int id_ = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = id_*N_PER_T;
  const int idE = min((int)pc_c.Area(),(id_+1)*N_PER_T);
  __shared__ Eigen::Matrix<float,7,1,Eigen::DontAlign> sum[BLK_SIZE];
  sum[tid] = Eigen::Matrix<float,7,1,Eigen::DontAlign>::Zero();
  for (int id=idS; id<idE; ++id) {
    const int idx = id%pc_c.w_;
    const int idy = id/pc_c.w_;
    // project current point into model frame to get association
    if (idx >= pc_c.w_ || idy >= pc_c.h_) continue;
    Vector3fda pc_ci = pc_c(idx,idy);
    Vector3fda pc_c_in_m = R_mc * pc_ci + t_mc;
    // project into model camera
    // TODO: doing the association the other way around might be more
    // stable since the model depth is smoothed
    Vector2fda x_c_in_m = cam.Project(pc_c_in_m);
    const int u = floor(x_c_in_m(0)+0.5f);
    const int v = floor(x_c_in_m(1)+0.5f);
    if (0 <= u && u < pc_c.w_ && 0 <= v && v < pc_c.h_
        && pc_ci(2) > 0. && pc_c_in_m(2) > 0.
        && IsValidData(pc_c_in_m)) {
      // found association -> check thresholds;
      Vector3fda n_c_in_m = R_mc * n_c(idx,idy);
      Vector3fda n_mi = n_m(u,v);
      const float dot  = n_mi.dot(n_c_in_m);
      if (dot > dotThr && IsValidData(n_mi)) {
        // association is good -> accumulate
        Eigen::Matrix<float,7,1,Eigen::DontAlign> upperTriangle;
        upperTriangle(0) = n_mi(0)*n_c_in_m(0);
        upperTriangle(1) = n_mi(1)*n_c_in_m(0);
        upperTriangle(2) = n_mi(2)*n_c_in_m(0);
        upperTriangle(3) = n_mi(1)*n_c_in_m(1);
        upperTriangle(4) = n_mi(1)*n_c_in_m(2);
        upperTriangle(5) = n_mi(2)*n_c_in_m(2);
        upperTriangle(6) = 1.; // to get number of data points
        sum[tid] += upperTriangle;
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
  if(tid < 7) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd(&out[tid], sum[0](tid)+sum[1](tid));
  }
}

void ICPStepRotation (
    Image<Vector3fda> n_m,
    Image<Vector3fda> n_c,
    Image<Vector3fda> pc_c,
    Matrix3fda R_mc, 
    Vector3fda t_mc, 
    const Camera<float>& cam,
    float dotThr,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& N,
    float& count
    ) {
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,pc_c.Area()/10,32);
  ManagedDeviceImage<float> out(7,1);
  cudaMemset(out.ptr_, 0, 7*sizeof(float));

  KernelICPStepRotation<32><<<blocks,threads>>>(n_m,n_c,pc_c,R_mc,t_mc,cam,
      dotThr,10,out);
  checkCudaErrors(cudaDeviceSynchronize());
  ManagedHostImage<float> nUpperTri(7,1);
  cudaMemcpy(nUpperTri.ptr_,out.ptr_,7*sizeof(float), cudaMemcpyDeviceToHost);

  //for (int i=0; i<29; ++i) std::cout << sumAb[i] << "\t";
  //std::cout << std::endl;
  N.fill(0.);
  int k = 0;
  for (int i=0; i<3; ++i) {
    for (int j=i; j<3; ++j) {
      float val = nUpperTri[k++];
      N(i,j) = val;
      N(j,i) = val;
    }
  }
  count = nUpperTri[6];
  //std::cout << ATA << std::endl << ATb.transpose() << std::endl;
  //std::cout << "\terror&count " << error << " " << count << std::endl;
}

}