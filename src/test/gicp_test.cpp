#include <cstdlib>
#include <iostream>
#include <gtest/gtest.h>

TEST(GICPTest, Alignment) {
  system("pwd");
  std::cerr << "hello" << std::endl;

  EXPECT_EQ(1, 1);
}

TEST(GICPTest, Alignment2) {
  std::cerr << "eee" << std::endl;
  EXPECT_EQ(1, 1);
}
