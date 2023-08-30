#include "test_lib.h"

#include <header_lib.h>
#include <iostream>
#include <sstream>

TEST_CASE("testing header_lib") {
  std::ostringstream oss;
  std::streambuf*    p_cout_streambuf = std::cout.rdbuf();
  std::cout.rdbuf(oss.rdbuf());

  header_p();

  std::cout.rdbuf(p_cout_streambuf);  // restore
  CHECK(oss.str() == "header_lib\n");
}

