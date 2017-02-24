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
#include <tdp/manifold/SO3.h>
#include <tdp/cuda/cuda.cuh>
//#include <tdp/icp/icp.cuh>
//#include <tdp/icp/photoSO3.h>

namespace tdp {

// T_mc: R_model_observation
template<int BLK_SIZE, int D, typename Derived>
__global__ void KernelSO3TextureStep(
    Image<float> grey_p,
    Image<float> grey_c,
    Image<Vector2fda> gradGrey_c,
    Image<Vector3fda> rays,
    SO3f R_cp, 
    CameraBase<float,D,Derived> cam,
    int N_PER_T,
    Image<float> out
    ) {
  assert(BLK_SIZE >=11);
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int idS = idx*N_PER_T;
  const int N = grey_p.w_*grey_p.h_;
  const int idE = min(N,(idx+1)*N_PER_T);

  SharedMemory<Vector11fda> smem;
  Vector11fda* sum = smem.getPointer();

  sum[tid] = Vector11fda::Zero();
  for (int id=idS; id<idE; ++id) {
    const int u = id%grey_p.w_;
    const int v = id/grey_p.w_;
    tdp::Vector3fda ray_c = R_cp*rays(u,v);
    tdp::Vector2fda x = cam.Project(ray_c);
    if (grey_p.Inside(x)) {
      float ab[4];      
      Eigen::Map<Vector3fda> Ai(&(ab[0]));
      Ai = -(R_cp.matrix()*SO3mat<float>::invVee(rays(u,v))).transpose()*
        cam.Jproject(ray_c).transpose() * gradGrey_c.GetBilinear(x);
      ab[3] = -grey_c.GetBilinear(x) + grey_p(u,v);
      Eigen::Matrix<float,11,1,Eigen::DontAlign> upperTriangle;
      int k=0;
#pragma unroll
      for (int i=0; i<4; ++i) {
        for (int j=i; j<4; ++j) {
          upperTriangle(k++) = ab[i]*ab[j];
        }
      }
      upperTriangle(10) = 1.; // to get number of data points
      sum[tid] += upperTriangle;
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
  if(tid < 11) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd(&out[tid], sum[0](tid)+sum[1](tid));
  }
}

template<int D, typename Derived>
void SO3TextureStep (
    Image<float> grey_p,
    Image<float> grey_c,
    Image<Vector2fda> gradGrey_c,
    Image<Vector3fda> rays,
    SO3f R_cp, 
    const CameraBase<float,D,Derived>& cam,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,3,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    ) {
  const size_t BLK_SIZE = 32;
  size_t N = grey_p.w_*grey_p.h_;
  dim3 threads, blocks;
  ComputeKernelParamsForArray(blocks,threads,N/2,BLK_SIZE);
  ManagedDeviceImage<float> out(11,1);
  cudaMemset(out.ptr_, 0, 11*sizeof(float));

  KernelSO3TextureStep<BLK_SIZE,D,Derived><<<blocks,threads,
    BLK_SIZE*sizeof(Vector11fda)>>>( grey_p, grey_c, gradGrey_c, rays,
        R_cp, cam, 2, out);
  checkCudaErrors(cudaDeviceSynchronize());
  ManagedHostImage<float> sumAb(11,1);
  cudaMemcpy(sumAb.ptr_,out.ptr_,11*sizeof(float), cudaMemcpyDeviceToHost);

  //for (int i=0; i<29; ++i) std::cout << sumAb[i] << "\t";
  //std::cout << std::endl;
  ATA.fill(0.);
  ATb.fill(0.);
  int k = 0;
  for (int i=0; i<3; ++i) {
    for (int j=i; j<4; ++j) {
      float val = sumAb[k++];
      if (j==3)  {
        ATb(i) = val;
      } else {
        ATA(i,j) = val;
        ATA(j,i) = val;
      }
    }
  }
  count = sumAb[11];
  error = sumAb[10]/count;
  //std::cout << ATA << std::endl << ATb.transpose() << std::endl;
  //std::cout << "\terror&count " << error << " " << count << std::endl;
}

template void SO3TextureStep (
    Image<float> grey_p,
    Image<float> grey_c,
    Image<Vector2fda> gradGrey_c,
    Image<Vector3fda> rays,
    SO3f R_cp, 
    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,3,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );
template void SO3TextureStep (
    Image<float> grey_p,
    Image<float> grey_c,
    Image<Vector2fda> gradGrey_c,
    Image<Vector3fda> rays,
    SO3f R_cp, 
    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
    Eigen::Matrix<float,3,3,Eigen::DontAlign>& ATA,
    Eigen::Matrix<float,3,1,Eigen::DontAlign>& ATb,
    float& error,
    float& count
    );

// explicit instantiation
//template void SO3TextureStep (
//    Image<float> grey_p,
//    Image<float> grey_c,
//    Image<Vector2fda> gradGrey_c,
//    Image<Vector3fda> rays,
//    SO3f R_cp, 
//    const CameraBase<float,Camera<float>::NumParams,Camera<float>>& cam,
//    Eigen::Matrix<float,3,3,Eigen::DontAlign>& ATA,
//    Eigen::Matrix<float,3,1,Eigen::DontAlign>& ATb,
//    float& error,
//    float& count
//    );
//template void SO3TextureStep (
//    Image<float> grey_p,
//    Image<float> grey_c,
//    Image<Vector2fda> gradGrey_c,
//    Image<Vector3fda> rays,
//    SO3f R_cp, 
//    const CameraBase<float,CameraPoly3<float>::NumParams,CameraPoly3<float>>& cam,
//    Eigen::Matrix<float,3,3,Eigen::DontAlign>& ATA,
//    Eigen::Matrix<float,3,1,Eigen::DontAlign>& ATb,
//    float& error,
//    float& count
//    );


}
