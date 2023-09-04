#include <vector>
#include <iostream>
#include <random>
#include "cuda_lib.h"

void PrintDeviceInfo();
void GenerateBgra8K(uint8_t* buffer, int dataSize);
void convertPixelFormatCpu(const uint8_t* inputBgra, uint8_t* outputYuv, int numPixels);
__global__ void convertPixelFormat(const uint8_t* inputBgra, uint8_t* outputYuv, int numPixels);

void Bgra2YuvTest() {
  PrintDeviceInfo();

  uint8_t* bgraBuffer{};
  uint8_t* yuvBuffer = nullptr;
  uint8_t* deviceBgraBuffer = nullptr;
  uint8_t* deviceYuvBuffer = nullptr;

  const int dataSizeBgra = 7680 * 4320 * 4;
  const int dataSizeYuv = 7680 * 4320 * 3;
  CUDA_CALL(cudaMallocHost(&bgraBuffer, dataSizeBgra));
  CUDA_CALL(cudaMallocHost(&yuvBuffer, dataSizeYuv));
  CUDA_CALL(cudaMalloc(&deviceBgraBuffer, dataSizeBgra));
  CUDA_CALL(cudaMalloc(&deviceYuvBuffer, dataSizeYuv));

  std::vector<uint8_t> yuvCpuBuffer(dataSizeYuv);

  cudaEvent_t start{};
  cudaEvent_t stop = nullptr;
  float elapsedTime{};
  float elapsedTimeTotal{};
  float dataRate{};
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

  std::cout << " " << std::endl;
  std::cout << "Generating 7680 x 4320 BRGA8888 image, data size: " << dataSizeBgra << std::endl;
  GenerateBgra8K(bgraBuffer, dataSizeBgra);

  std::cout << " " << std::endl;
  std::cout << "Computing results using CPU." << std::endl;
  std::cout << " " << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  convertPixelFormatCpu(bgraBuffer, yuvCpuBuffer.data(), 7680 * 4320);
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "    Whole process took " << elapsedTime << "ms." << std::endl;

  std::cout << " " << std::endl;
  std::cout << "Computing results using GPU, default stream." << std::endl;
  std::cout << " " << std::endl;

  std::cout << "    Move data to GPU." << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  CUDA_CALL(cudaMemcpy(deviceBgraBuffer, bgraBuffer, dataSizeBgra, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  dataRate = dataSizeBgra / (elapsedTime / 1000.0) / 1.0e9;
  elapsedTimeTotal = elapsedTime;
  std::cout << "        Data transfer took " << elapsedTime << "ms." << std::endl;
  std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

  std::cout << "    Convert 8-bit BGRA to 8-bit YUV." << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  convertPixelFormat<<<32400, 1024>>>(deviceBgraBuffer, deviceYuvBuffer, 7680 * 4320);
  CUDA_CHECK();
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  dataRate = dataSizeBgra / (elapsedTime / 1000.0) / 1.0e9;
  elapsedTimeTotal += elapsedTime;
  std::cout << "        Processing of 8K image took " << elapsedTime << "ms." << std::endl;
  std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

  std::cout << "    Move data to CPU." << std::endl;
  CUDA_CALL(cudaEventRecord(start, 0));
  CUDA_CALL(cudaMemcpy(yuvBuffer, deviceYuvBuffer, dataSizeYuv, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  dataRate = dataSizeYuv / (elapsedTime / 1000.0) / 1.0e9;
  elapsedTimeTotal += elapsedTime;
  std::cout << "        Data transfer took " << elapsedTime << "ms." << std::endl;
  std::cout << "        Performance is " << dataRate << "GB/s." << std::endl;

  std::cout << "    Whole process took " << elapsedTimeTotal << "ms." << std::endl;

  std::cout << "    Compare CPU and GPU results ..." << std::endl;
  bool foundMistake = false;
  for (int i = 0; i < dataSizeYuv; i++) {
    if (yuvCpuBuffer[i] != yuvBuffer[i]) {
      foundMistake = true;
      break;
    }
  }

  if (foundMistake) {
    std::cout << "        Results are NOT the same." << std::endl;
  } else {
    std::cout << "        Results are the same." << std::endl;
  }

  const int nStreams = 16;

  std::cout << " " << std::endl;
  std::cout << "Computing results using GPU, using " << nStreams << " streams." << std::endl;
  std::cout << " " << std::endl;

  cudaStream_t streams[nStreams];
  std::cout << "    Creating " << nStreams << " CUDA streams." << std::endl;
  for (auto& stream : streams) {
    CUDA_CALL(cudaStreamCreate(&stream));
  }

  int brgaOffset = 0;
  int yuvOffset = 0;
  const int brgaChunkSize = dataSizeBgra / nStreams;
  const int yuvChunkSize = dataSizeYuv / nStreams;

  CUDA_CALL(cudaEventRecord(start, 0));
  for (int i = 0; i < nStreams; i++) {
    std::cout << "        Launching stream " << i << "." << std::endl;
    brgaOffset = brgaChunkSize * i;
    yuvOffset = yuvChunkSize * i;
    CUDA_CALL(cudaMemcpyAsync(deviceBgraBuffer + brgaOffset,
                              bgraBuffer + brgaOffset,
                              brgaChunkSize,
                              cudaMemcpyHostToDevice,
                              streams[i]));

    convertPixelFormat<<<4096, 1024, 0, streams[i]>>>(deviceBgraBuffer + brgaOffset, deviceYuvBuffer + yuvOffset, brgaChunkSize / 4);

    CUDA_CALL(cudaMemcpyAsync(yuvBuffer + yuvOffset,
                              deviceYuvBuffer + yuvOffset,
                              yuvChunkSize,
                              cudaMemcpyDeviceToHost,
                              streams[i]));
  }

  CUDA_CHECK();
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaEventRecord(stop, 0));
  CUDA_CALL(cudaEventSynchronize(stop));
  CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "    Whole process took " << elapsedTime << "ms." << std::endl;

  std::cout << "    Compare CPU and GPU results ..." << std::endl;
  for (int i = 0; i < dataSizeYuv; i++) {
    if (yuvCpuBuffer[i] != yuvBuffer[i]) {
      foundMistake = true;
      break;
    }
  }

  if (foundMistake) {
    std::cout << "        Results are NOT the same." << std::endl;
  } else {
    std::cout << "        Results are the same." << std::endl;
  }

  CUDA_CALL(cudaFreeHost(bgraBuffer));
  CUDA_CALL(cudaFreeHost(yuvBuffer));
  CUDA_CALL(cudaFree(deviceBgraBuffer));
  CUDA_CALL(cudaFree(deviceYuvBuffer));
}

void PrintDeviceInfo() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Number of device(s): " << deviceCount << std::endl;
  if (deviceCount == 0) {
    std::cout << "There is no device supporting CUDA" << std::endl;
    return;
  }

  cudaDeviceProp info{};
  for (int i = 0; i < deviceCount; i++) {
    cudaGetDeviceProperties(&info, i);
    std::cout << "Device " << i << std::endl;
    std::cout << "    Name:                    " << std::string(info.name) << std::endl;
    std::cout << "    Glocbal memory:          " << info.totalGlobalMem / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "    Shared memory per block: " << info.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "    Warp size:               " << info.warpSize << std::endl;
    std::cout << "    Max thread per block:    " << info.maxThreadsPerBlock << std::endl;
    std::cout << "    Thread dimension limits: " << info.maxThreadsDim[0] << " x "
              << info.maxThreadsDim[1] << " x "
              << info.maxThreadsDim[2] << std::endl;
    std::cout << "    Max grid size:           " << info.maxGridSize[0] << " x "
              << info.maxGridSize[1] << " x "
              << info.maxGridSize[2] << std::endl;
    std::cout << "    Compute capability:      " << info.major << "." << info.minor << std::endl;
  }
}

void GenerateBgra8K(uint8_t* buffer, int dataSize) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sampler(0, 255);

  for (int i = 0; i < dataSize / 4; i++) {
    buffer[i * 4] = sampler(gen);
    buffer[i * 4 + 1] = sampler(gen);
    buffer[i * 4 + 2] = sampler(gen);
    buffer[i * 4 + 3] = 255;
  }
}

void convertPixelFormatCpu(const uint8_t* inputBgra, uint8_t* outputYuv, int numPixels) {
  short3 yuv16;
  char3 yuv8;
  for (int idx = 0; idx < numPixels; idx++) {
    yuv16.x = 66 * inputBgra[idx * 4 + 2] + 129 * inputBgra[idx * 4 + 1] + 25 * inputBgra[idx * 4];
    yuv16.y = -38 * inputBgra[idx * 4 + 2] + -74 * inputBgra[idx * 4 + 1] + 112 * inputBgra[idx * 4];
    yuv16.z = 112 * inputBgra[idx * 4 + 2] + -94 * inputBgra[idx * 4 + 1] + -18 * inputBgra[idx * 4];

    yuv8.x = (yuv16.x >> 8) + 16;
    yuv8.y = (yuv16.y >> 8) + 128;
    yuv8.z = (yuv16.z >> 8) + 128;

    *(reinterpret_cast<char3*>(&outputYuv[idx * 3])) = yuv8;
  }
}

__global__ void convertPixelFormat(const uint8_t* inputBgra, uint8_t* outputYuv, int numPixels) {
  int stride = gridDim.x * blockDim.x;              // total threads
  int idx = threadIdx.x + blockIdx.x * blockDim.x;  // thread id
  short3 yuv16;
  char3 yuv8;

  while (idx <= numPixels) {
    if (idx < numPixels) {
      yuv16.x = 66 * inputBgra[idx * 4 + 2] + 129 * inputBgra[idx * 4 + 1] + 25 * inputBgra[idx * 4];
      yuv16.y = -38 * inputBgra[idx * 4 + 2] + -74 * inputBgra[idx * 4 + 1] + 112 * inputBgra[idx * 4];
      yuv16.z = 112 * inputBgra[idx * 4 + 2] + -94 * inputBgra[idx * 4 + 1] + -18 * inputBgra[idx * 4];

      yuv8.x = (yuv16.x >> 8) + 16;
      yuv8.y = (yuv16.y >> 8) + 128;
      yuv8.z = (yuv16.z >> 8) + 128;

      *(reinterpret_cast<char3*>(&outputYuv[idx * 3])) = yuv8;
    }
    idx += stride;
  }
}
