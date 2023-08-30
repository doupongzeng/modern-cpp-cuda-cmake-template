project(project_nanobench CXX)

include(FetchContent)
FetchContent_Declare(
     nanobench
     GIT_REPOSITORY https://github.com/martinus/nanobench.git
     GIT_TAG v4.1.0
     GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nanobench)

file(
  GLOB_RECURSE TEST_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE header_lib static_lib shared_lib nanobench)
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
