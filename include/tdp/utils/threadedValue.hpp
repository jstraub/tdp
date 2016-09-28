
#pragma once

#include <thread>
#include <mutex>

namespace tdp {

template <typename T>
class ThreadedValue {
 public:
  ThreadedValue() {}
  ThreadedValue(T val) : val_(val) {}
  ~ThreadedValue() {}

  T Get() {
    std::lock_guard<std::mutex> lock(mut_);
    return val_;
  }
  void Set(T val) {
    std::lock_guard<std::mutex> lock(mut_);
    val_ = val;
  }

  void Increment() {
    std::lock_guard<std::mutex> lock(mut_);
    val_++;
  }

 private:
  T val_;
  std::mutex mut_;
};

}
