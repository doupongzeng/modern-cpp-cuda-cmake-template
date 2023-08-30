#include "test_lib.h"

#include <iostream>
#include <sstream>
#include <static_lib.h>

TEST_CASE("testing header_lib") {
  std::ostringstream oss;
  std::streambuf*    p_cout_streambuf = std::cout.rdbuf();
  std::cout.rdbuf(oss.rdbuf());

  static_lib_p();

  std::cout.rdbuf(p_cout_streambuf);  // restore
  CHECK(oss.str() == "static_lib_p\n");
}
