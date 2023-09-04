#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef DEBUG
#define CUDA_CALL(F)                                                      \
  if ((F) != cudaSuccess) {                                               \
    printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
           __FILE__, __LINE__);                                           \
    exit(-1);                                                             \
  }
#define CUDA_CHECK()                                                      \
  if ((cudaPeekAtLastError()) != cudaSuccess) {                           \
    printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
           __FILE__, __LINE__ - 1);                                       \
    exit(-1);                                                             \
  }
#else
#define CUDA_CALL(F) (F)
#define CUDA_CHECK()
#endif

void Bgra2YuvTest();
void histogramTest(size_t img_w, size_t img_h, unsigned char* data, unsigned char* data1);

