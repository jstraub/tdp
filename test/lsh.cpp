#include <tdp/testing/testing.h>
#include <tdp/features/lsh.h>

using namespace tdp;

TEST(lsh, init) {
  LSH<14> lsh;
  lsh.PrintHash();

  Brief brief;
  for (size_t i=0; i<100; ++i) {
    brief.desc_ = Vector8uda::Random();
    brief.frame_ = 0;
    lsh.Insert(brief);
  }
  for (size_t i=0; i<100; ++i) {
    brief.desc_ = Vector8uda::Random();
    brief.frame_ = 1;
    lsh.Insert(brief);
  }
  brief.desc_ = Vector8uda::Random();
  brief.frame_ = 1;
  if(lsh.SearchBest(brief)) {
    std::cout << "found" << std::endl;
  } else {
    std::cout << "miss" << std::endl;
  }
}

TEST(lshforest, init) {
  LshForest<14> lsh(11);
  lsh.PrintHashs();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
