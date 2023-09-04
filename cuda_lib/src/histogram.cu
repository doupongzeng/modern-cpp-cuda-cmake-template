#include <cstddef>
#include <iostream>
#include <chrono>
#include "cuda_lib.h"

__global__ void clearSubHistograms(
    uint32_t* subHistogram,
    size_t nbins,
    size_t nsubhist,
    size_t subHistogramPitch) {
  uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t step = blockDim.x * gridDim.x;

  for (int i = sid; i < nbins; i += step) {
    for (int j = 0; j < nsubhist; j++) {
      subHistogram[j * subHistogramPitch + i] = 0;
    }
  }
}

__global__ void subhist(const uint8_t* image, size_t w, size_t h, size_t memPitch,
                        uint32_t* subHistogram, size_t nbins, size_t nsubhist, size_t subHistogramPitch) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t subHisId = blockIdx.x;
  if (idx < w) {
    for (int i = 0; i < h; ++i) {
      uint8_t pv = image[i * memPitch + idx];
      atomicAdd(&subHistogram[subHisId * subHistogramPitch + pv], 1);
    }
  }
}

__global__ void subhist_shared_memory(const uint8_t* image, size_t w, size_t h, size_t memPitch,
                                      uint32_t* subHistogram, size_t nbins, size_t nsubhist, size_t subHistogramPitch) {
  // shared memory (smem) 被同一个block中的所有thread共享。
  // 每个block的smem大小在kernel launch 的时候确定。
  // kernel<<<GridDim, BlockDim, smem_size>>>(...)
  extern __shared__ uint32_t localHist[];

  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t sid = threadIdx.x;
  for (int i = sid; i < nbins; i += blockDim.x) {
    localHist[i] = 0;
  }
  __syncthreads();

  if (idx < w) {
    for (int i = 0; i < h; ++i) {
      uint8_t pv = image[i * memPitch + idx];
      atomicAdd(&localHist[pv], 1);
    }
  }
  __syncthreads();

  uint32_t subHisId = blockIdx.x;
  for (int i = sid; i < nbins; i += blockDim.x) {
    subHistogram[subHisId * subHistogramPitch + i] = localHist[i];
  }
}

const size_t kYFactor = 8;
__global__ void subhist_optimiz_launch_config_size(const uint8_t* image, size_t w, size_t h, size_t memPitch,
                                                   uint32_t* subHistogram, size_t nbins, size_t nsubhist, size_t subHistogramPitch) {
  extern __shared__ uint32_t localHist[];

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  uint32_t xstride = gridDim.x * blockDim.x;
  uint32_t ystride = gridDim.y * blockDim.y * kYFactor;

  uint32_t sid = threadIdx.x;
  for (int i = sid; i < nbins; i += blockDim.x) {
    localHist[i] = 0;
  }
  __syncthreads();
  for (int j = idx; j < w; j += xstride) {
    for (int i = idy * kYFactor; i < h; i += ystride) {
      // 在y方向上每个thread处理8个像素
      for (int k = 0; k < min(kYFactor, h - i); k++) {
        uint8_t pv = image[(i + k) * memPitch + j];
        atomicAdd(&localHist[pv], 1);
      }
    }
  }
  __syncthreads();
  uint32_t subHisId = idy * gridDim.x + blockIdx.x;
  for (int i = sid; i < nbins; i += blockDim.x) {
    subHistogram[subHisId * subHistogramPitch + i] = localHist[i];
  }
}

__global__ void sumSubHistograms(const uint32_t* subHistogram, size_t nbins, size_t nsubhist,
                                 size_t subHistogramPitch, uint32_t* histogram) {
  uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t step = blockDim.x * gridDim.x;

  for (int i = sid; i < nbins; i += step) {
    uint32_t sum = 0;
    for (int j = 0; j < nsubhist; ++j) {
      sum += subHistogram[j * subHistogramPitch + i];
    }
    histogram[i] = sum;
  }
}

void histogramTest(size_t img_w, size_t img_h, unsigned char* data, unsigned char* data1) {
  std::chrono::time_point<std::chrono::steady_clock> t1, t2;
  int device{};
  cudaDeviceProp properties = {};

  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);
  std::cout << "device: " << device << std::endl;
  std::cout << "        multiProcessorCount: " << properties.multiProcessorCount << std::endl;
  std::cout << "maxThreadsPerMultiProcessor: " << properties.maxThreadsPerMultiProcessor << std::endl;

  size_t nbins = 256;
  size_t width = img_w;
  size_t height = img_h;
  size_t npix = width * height;

  uint8_t* inputDevice{};
  uint8_t* outputDevice{};

  uint32_t* histogram{};
  uint32_t* histogramHost{};
  uint32_t* subHistogram{};

  size_t inputPitch{};
  size_t outputPitch{};
  size_t subHistogramPitch{};

  size_t bsx = 512;
  size_t gsx = 1;
  size_t gsy = properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor / bsx;
  size_t smem = sizeof(uint32_t) * nbins;
  size_t nsubhist = gsx * gsy;

  dim3 subHistGridDim(gsx, gsy, 1);
  dim3 subHistBlockDim(bsx, 1, 1);
  dim3 sumGridDim((nbins + bsx - 1) / bsx, 1, 1);
  dim3 sumBlockDim(bsx, 1, 1);

  std::cout << "image width: " << width << std::endl;
  std::cout << "image height: " << height << std::endl;

  cudaMallocPitch(&inputDevice, &inputPitch, width, height);
  cudaMallocPitch(&outputDevice, &outputPitch, width, height);
  cudaMallocPitch(&subHistogram, &subHistogramPitch, sizeof(uint32_t) * nbins, nsubhist);
  cudaMallocHost(&histogramHost, sizeof(uint32_t) * nbins);
  cudaMalloc(&histogram, sizeof(uint32_t) * nbins);
  cudaHostRegister(data, sizeof(uint8_t) * npix, cudaHostRegisterDefault);
  cudaHostRegister(data1, sizeof(uint8_t) * npix, cudaHostRegisterDefault);
  cudaMemcpy2D(inputDevice, inputPitch, data, width, width, height, cudaMemcpyHostToDevice);

  double durationSum = 0.0;
  for (int i = 0; i < 100; i++) {
    t1 = std::chrono::steady_clock::now();
    subhist_optimiz_launch_config_size<<<subHistGridDim, subHistBlockDim, smem>>>(inputDevice, width, height, inputPitch, subHistogram, nbins, nsubhist, subHistogramPitch);
    sumSubHistograms<<<sumGridDim, sumBlockDim>>>(subHistogram, nbins, nsubhist, subHistogramPitch, histogram);
    cudaDeviceSynchronize();
    t2 = std::chrono::steady_clock::now();
    uint64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    durationSum += duration;
  }

  std::cout << "Average GPU time: " << durationSum / 100.f << std::endl;

  cudaMemcpy2D(data1, width, inputDevice, inputPitch, width, height, cudaMemcpyDeviceToHost);
  cudaMemcpy(histogramHost, histogram, sizeof(uint32_t) * nbins, cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < nbins; i++) {
    // if(histogramHost[i] != (uint32_t)cvH)
  }
  cudaFree(inputDevice);
  cudaFree(outputDevice);
  cudaFree(subHistogram);
  cudaFreeHost(histogramHost);
  cudaFree(histogram);
  cudaHostUnregister(data);
  cudaHostUnregister(data1);
}

