#include <nanobench.h>
#include <shared_lib.h>

int main() {
  ankerl::nanobench::Bench().run("shared_lib", [&] {
    shared_lib_p();
  });
  return 0;
}
