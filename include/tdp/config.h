#pragma once 

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int32_t

#ifdef __CUDACC__ 
#  define TDP_HOST_DEVICE __host__ __device__
#else
#  define TDP_HOST_DEVICE
#endif
