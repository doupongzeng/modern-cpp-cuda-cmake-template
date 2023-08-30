#include <cstdio>
#include "cuda_lib.h"
#include <cuda_runtime.h>

__global__ void CudaLibFuncImpl() { printf("Hello my CUDA library\n"); }

void CudaLibFunc()
{
    CudaLibFuncImpl<<<1, 1>>>();
    cudaDeviceSynchronize();
}
