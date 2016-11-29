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
#include <tdp/icp/icp.cuh>

namespace tdp {

// T_mc: R_model_observation
template<int BLK_SIZE>
__global__ void KernelICPStep(
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    Image<int> assoc_om,
    SE3f T_mo, 
    float dotThr,
    float distThr,
    int N_PER_T,
    Image<float> out
    ) {
  assert(BLK_SIZE >=29);
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = pc_m.w_*pc_m.h_;
  const int idE = min(N,(idx+1)*N_PER_T);

  SharedMemory<Vector29fda> smem;
  Vector29fda* sum = smem.getPointer();

  sum[tid] = Vector29fda::Zero();
  for (int id=idS; id<idE; ++id) {
    const int x = id%pc_m.w_;
    const int y = id/pc_m.w_;
    if (x<pc_m.w_ && y<pc_m.h_) {
      const int id_o = assoc_om(x,y);
      const int u = id_o%pc_o.w_;
      const int v = id_o/pc_o.w_;
      if (0<=u && u<pc_o.w_ && 0<=v && v<pc_o.h_) {
        // found association -> check thresholds;
        Vector3fda n_o_in_m = T_mo.rotation()*n_o(u,v);
        Vector3fda n_mi = n_m(x,y);
        Vector3fda pc_mi = pc_m(x,y);
        Vector3fda pc_oi = pc_o(u,v);
        Vector3fda pc_o_in_m = T_mo * pc_oi;
        const float dot  = n_mi.dot(n_o_in_m);
        const float dist = (pc_mi-pc_o_in_m).norm();
        if (dot > dotThr && dist < distThr && IsValidData(pc_mi)) {
          // association is good -> accumulate
          // if we found a valid association accumulate the A and b for A x = b
          // where x \in se{3} as well as the residual error
          float ab[7];      
          Eigen::Map<Vector3fda> top(&(ab[0]));
          Eigen::Map<Vector3fda> bottom(&(ab[3]));
          // as in mp3guy: 
          top = (pc_o_in_m).cross(n_mi);
          bottom = n_mi;
          ab[6] = n_mi.dot(pc_mi-pc_o_in_m);
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
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    Image<int> assoc_om,
    const SE3f& T_mo, 
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    ) {
  const size_t BLK_SIZE = 32;
  size_t N = pc_m.w_*pc_m.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,BLK_SIZE);
  ManagedDeviceImage<float> out(29,1);
  cudaMemset(out.ptr_, 0, 29*sizeof(float));

  KernelICPStep<BLK_SIZE><<<blocks,threads,
    BLK_SIZE*sizeof(Vector29fda)>>>(
        pc_m,n_m,pc_o,n_o,assoc_om,T_mo,dotThr,distThr,10,out);
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

// T_mc: R_model_observation
template<int BLK_SIZE, int D, typename Derived>
__global__ void KernelICPStep(
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    SE3f T_mo, 
    SE3f T_co, 
    const CameraBase<float,D,Derived> cam,
    float dotThr,
    float distThr,
    int N_PER_T,
    Image<float> out
    ) {
  assert(BLK_SIZE >=29);
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = pc_m.w_*pc_m.h_;
  const int idE = min(N,(idx+1)*N_PER_T);

  SharedMemory<Vector29fda> smem;
  Vector29fda* sum = smem.getPointer();

  sum[tid] = Vector29fda::Zero();
  for (int id=idS; id<idE; ++id) {
    const int x = id%pc_o.w_;
    const int y = id/pc_o.w_;
    int u, v;
    //printf("%d %d\n",x,y);
    int res = AssociateModelIntoCurrent<D,Derived>(x, y, pc_m, T_mo,
        T_co, cam, u, v);
    //printf("%d %d %d\n",x,y,res);
    if (res == 0) {
      //printf("%d %d %d %d\n",x,y,u,v);
      // found association -> check thresholds;
      Vector3fda n_o_in_m = T_mo.rotation()*n_o(u,v);
      Vector3fda n_mi = n_m(x,y);
      Vector3fda n_m_in_oi = T_mo.rotation().Inverse()*n_mi;
      Vector3fda pc_mi = pc_m(x,y);
      Vector3fda pc_oi = pc_o(u,v);
      Vector3fda pc_o_in_m = T_mo * pc_oi;
      const float dot  = n_mi.dot(n_o_in_m);
      const float dist = (pc_mi-pc_o_in_m).norm();
      if (dot > dotThr && dist < distThr && IsValidData(pc_mi)) {
        // association is good -> accumulate
        // if we found a valid association accumulate the A and b for A x = b
        // where x \in se{3} as well as the residual error
        float ab[7];      
        Eigen::Map<Vector3fda> top(&(ab[0]));
        Eigen::Map<Vector3fda> bottom(&(ab[3]));
        // as in mp3guy:  (left multiplication of error)
//        top = (pc_o_in_m).cross(n_mi);
//        bottom = n_mi;
        // right multiplication of error
        top = (pc_oi).cross(n_m_in_oi);
        bottom = n_m_in_oi;
        float nTt = n_mi.dot(T_mo.translation());
//        top.array() += nTt;
//        bottom.array() += nTt;
        ab[6] = n_mi.dot(pc_mi-pc_o_in_m);
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

template<int D, typename Derived>
void ICPStep (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    const SE3f& T_mo, 
    const SE3f& T_cm,
    const CameraBase<float,D,Derived>& cam,
    float dotThr,
    float distThr,
    Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    ) {
  const size_t BLK_SIZE = 32;
  size_t N = pc_m.w_*pc_m.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,BLK_SIZE);
  ManagedDeviceImage<float> out(29,1);
  cudaMemset(out.ptr_, 0, 29*sizeof(float));

  KernelICPStep<BLK_SIZE,D,Derived><<<blocks,threads,
    BLK_SIZE*sizeof(Vector29fda)>>>(
        pc_m,n_m,pc_o,n_o,T_mo,T_cm,cam,
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

// explicit instantiation
template void ICPStep (
    Image<Vector3fda> pc_m, Image<Vector3fda> n_m, Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o, const SE3f& T_mo, const SE3f& T_cm,
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    float dotThr, float distThr, Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb, float& error, float& count);
template void ICPStep (
    Image<Vector3fda> pc_m, Image<Vector3fda> n_m, Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o, const SE3f& T_mo, const SE3f& T_cm,
    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
    float dotThr, float distThr, Eigen::Matrix<float,6,6,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,6,1,Eigen::DontAlign>& ATb, float& error, float& count);


// T_mc: R_model_observation
template<int BLK_SIZE, int D, class Derived>
__global__ void KernelICPVisualizeAssoc(
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    SE3f T_mo,
    const CameraBase<float,D,Derived> cam,
    float dotThr,
    float distThr,
    int N_PER_T,
    Image<float> assoc_m,
    Image<float> assoc_o
    ) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = pc_m.w_*pc_m.h_;
  const int idE = min(N,(idx+1)*N_PER_T);

  for (int id=idS; id<idE; ++id) {
    const int x = id%pc_m.w_;
    const int y = id/pc_m.w_;
    int u, v;
    int res = AssociateModelIntoCurrent<D,Derived>(x, y, pc_m, T_mo,
        tdp::SE3f(), cam, u, v);
    if (res == 0) {
      // found association -> check thresholds;
      Vector3fda pc_mi = pc_m(x,y);
      Vector3fda n_mi = n_m(x,y);
      Vector3fda n_o_in_m = T_mo.rotation() * n_o(u,v);
      Vector3fda pc_oi = pc_o(u,v);
      Vector3fda pc_o_in_m = T_mo * pc_oi;
      float dot  = n_mi.dot(n_o_in_m);
      float dist = (pc_mi-pc_o_in_m).norm();
      if (dot > dotThr && dist < distThr && IsValidData(pc_mi)) {
        // association is good -> accumulate
        //assoc_m(u,v) = n_mi.dot(-pc_mi+pc_o_in_m);
        //assoc_o(x,y) = n_mi.dot(-pc_mi+pc_o_in_m);
        assoc_m(x,y) = n_mi.dot(-pc_mi+pc_o_in_m);
        //        assoc_o(u,v) = n_mi.dot(-pc_mi+pc_o_in_m);
        //          if (threadIdx.x < 3) printf("%d,%d and %d,%d\n", x,y,u,v);
      }
    } else if (res < 3) {
      assoc_o(x,y) = res;
    }
  }
}

template<int D, typename Derived>
void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m,
    Image<Vector3fda> n_m,
    Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o,
    const SE3f& T_mo,
    const CameraBase<float,D,Derived>& cam,
    float angleThr,
    float distThr,
    Image<float>& assoc_m,
    Image<float>& assoc_o
    ) {
  size_t N = pc_m.w_*pc_m.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/10,256);
  KernelICPVisualizeAssoc<256,D,Derived><<<blocks,threads>>>(pc_m,n_m,pc_o,n_o,
      T_mo,cam, cos(angleThr*M_PI/180.),distThr,10,assoc_m, assoc_o);
  checkCudaErrors(cudaDeviceSynchronize());
}

template void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m, Image<Vector3fda> n_m, Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o, const SE3f& T_mo,
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    float angleThr, float distThr, Image<float>& assoc_m, Image<float>& assoc_o);
template void ICPVisualizeAssoc (
    Image<Vector3fda> pc_m, Image<Vector3fda> n_m, Image<Vector3fda> pc_o,
    Image<Vector3fda> n_o, const SE3f& T_mo,
    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
    float angleThr, float distThr, Image<float>& assoc_m, Image<float>& assoc_o);


}
