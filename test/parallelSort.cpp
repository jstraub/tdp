#include <tdp/testing/testing.h>
#include <tdp/utils/timer.hpp>
#include <tdp/sorts/parallelSorts.h>

#include <stdlib.h>
#include <algorithm>
#include <functional>

void runIntegerTest(size_t size, std::function<void(uint32_t, int*)> sortFunction) {
  int* values = new int[size];
  int* copies = new int[size];

  // Randomly initialize the arrays
  for (size_t i = 0; i < size; i++) {
    values[i] = rand() % 500;
    copies[i] = values[i];
  }

  sortFunction(size, values);
  std::sort(copies, copies + size);

  bool passed = true;
  for (size_t i = 0; i < size; i++) {
    if (values[i] != copies[i]) {
      passed = false;
    }
  }

  EXPECT_TRUE(passed);
}

TEST(bitonicSort, sortPowerOf2) {
  runIntegerTest(128, tdp::ParallelSorts<int>::bitonicSort);
}

TEST(bitonicSort, sortNotPowerOf2) {
  runIntegerTest(127, tdp::ParallelSorts<int>::bitonicSort);
}

TEST(oddEvenMergeSort, sortPowerOf2) {
  runIntegerTest(128, tdp::ParallelSorts<int>::oddEvenMergeSort);
}

TEST(oddEvenMergeSort, sortNotPowerOf2) {
  runIntegerTest(127, tdp::ParallelSorts<int>::oddEvenMergeSort);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

