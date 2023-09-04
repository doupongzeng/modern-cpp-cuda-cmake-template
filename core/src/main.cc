#include <opencv2/core/hal/interface.h>
#include "cuda_lib.h"
#include <iostream>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <cuda_runtime.h>

int test(int argc, char** argv) {

  std::chrono::time_point<std::chrono::steady_clock> t1;
  std::chrono::time_point<std::chrono::steady_clock> t2;

  cv::Mat image;
  cv::Mat image1;
  cv::Mat cvHist;

  image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  image1 = cv::Mat::zeros(image.rows, image.cols, image.depth());

  float range[] = {0, 256};
  const float* histRange = {range};
  int channles = 0;
  int hs = 20;

  t1 = std::chrono::steady_clock::now();
  cv::calcHist(&image, 1, &channles, cv::Mat(), cvHist, 1, &hs, &histRange, true, false);
  t2 = std::chrono::steady_clock::now();
  uint64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  std::cout << "CPU compute time: " << duration / 1000.f << "us" << std::endl;

  cv::namedWindow("Display window");
  cv::imshow("Display window", image);
  cv::waitKey(0);

  if (image.depth() != CV_8U) {
    std::cout << "incompatible data type" << std::endl;
    return 0;
  }

  histogramTest(image.cols, image.rows, image.data, image1.data);

  return 0;
}

int main(int argc, char** argv) {
  Bgra2YuvTest();
  test(argc, argv);
  return 0;
}

