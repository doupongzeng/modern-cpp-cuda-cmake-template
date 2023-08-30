cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(header_lib CXX)

add_library(header_lib INTERFACE)

# Boost
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_MULTITHREADED ON)
# find_package(Boost 1.65.0.0 REQUIRED)
target_compile_definitions(
  header_lib INTERFACE HEADER_LIB)
# target_include_directories(header_lib SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})

# header
target_include_directories(header_lib
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
