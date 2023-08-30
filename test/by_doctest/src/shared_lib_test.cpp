#include "test_lib.h"

#include <iostream>
#include <shared_lib.h>
#include <sstream>

TEST_CASE("testing header_lib") {
  std::ostringstream oss;
  std::streambuf*    p_cout_streambuf = std::cout.rdbuf();
  std::cout.rdbuf(oss.rdbuf());

  shared_lib_p();

  std::cout.rdbuf(p_cout_streambuf);  // restore
  CHECK(oss.str() == "shared_lib_p\n");
}
