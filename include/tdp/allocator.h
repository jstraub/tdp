#pragma once

namespace tdp {

template<typename T>
class CpuAllocator {
 public:
  ~CpuAllocator() {}

  static T* construct(size_t N) {
    T* ptr = new T[N];
    return ptr;
  }

  static void destroy(T* ptr) {
    delete[] ptr;
  }

 private:
  CpuAllocator() {}
};

}
