#include "cuda_lib.h"
#include "header_lib.h"
#include "shared_lib.h"
#include "static_lib.h"
#include "spdlog/spdlog.h"

#include <iostream>

#ifdef  HEADER_LIB

#endif

int main() {
  CudaLibFunc();
  shared_lib_p();
  static_lib_p();
  header_p();
  spdlog::info("hello spd log");
  std::cout << "main" << std::endl;

  return 0;
}
