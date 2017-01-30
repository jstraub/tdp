#include <tdp/testing/testing.h>
#include <tdp/utils/timer.hpp>
#include <tdp/sorts/parallel_sorts.h>

#include <stdlib.h>
#include <algorithm>

void runIntegerTest(size_t size) {
  int* values = new int[size];
  int* copies = new int[size];

  // Randomly initialize the arrays
  for (size_t i = 0; i < size; i++) {
    values[i] = rand() % 500;
    copies[i] = values[i];
  }

  tdp::ParallelSorts<int>::bitonicSort(size, values);
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
  runIntegerTest(128);
}

TEST(bitonicSort, sortNotPowerOf2) {
  runIntegerTest(255);
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
