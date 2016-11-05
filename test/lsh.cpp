#include <tdp/testing/testing.h>
#include <tdp/features/lsh.h>

using namespace tdp;

TEST(lsh, init) {
  ManagedLshForest<14> lsh(1);
  lsh.PrintHashs();

  Brief brief, query;
  for (size_t i=0; i<100; ++i) {
    brief.desc_ = Vector8uda::Random();
    brief.frame_ = 0;
    lsh.Insert(&brief);
  }
  std::vector<Brief> queries;
  for (size_t i=0; i<10; ++i) {
    brief.desc_ = Vector8uda::Random();
    brief.frame_ = 1;
    lsh.Insert(&brief);
    queries.push_back(brief);
  }
  lsh.PrintFillStatus();
  for (auto& query : queries) {
    Brief* res;
    int dist = 0;
    if(lsh.SearchBest(query,dist,res)) {
      std::cout << "found " << dist << std::endl;
    } else {
      std::cout << "miss" << std::endl;
    }
  }
}

TEST(lshforest, init) {
  ManagedLshForest<14> lsh(11);
  lsh.PrintHashs();

  Brief brief, query;
  for (size_t i=0; i<100; ++i) {
    brief.desc_ = Vector8uda::Random();
    brief.frame_ = 0;
    lsh.Insert(&brief);
  }
  std::vector<Brief> queries;
  for (size_t i=0; i<10; ++i) {
    brief.desc_ = Vector8uda::Random();
    brief.frame_ = 1;
    lsh.Insert(&brief);
    queries.push_back(brief);
  }
  lsh.PrintFillStatus();
  for (auto& query : queries) {
    Brief* res;
    int dist = 0;
    if(lsh.SearchBest(query,dist,res)) {
      std::cout << "found " << dist << std::endl;
    } else {
      std::cout << "miss" << std::endl;
    }
  }

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
