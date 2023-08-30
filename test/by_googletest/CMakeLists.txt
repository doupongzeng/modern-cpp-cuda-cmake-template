project(projec_test CXX)

# Google Test
find_package(GTest CONFIG REQUIRED)

file(
  GLOB_RECURSE TEST_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE header_lib static_lib shared_lib
  GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main)

# add tests
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
