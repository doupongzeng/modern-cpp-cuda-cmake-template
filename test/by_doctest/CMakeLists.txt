project(project_doctest CXX)

# set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
# set(CMAKE_CXX_VISIBILITY_PRESET default)
# fetch doctest from github
include(FetchContent)
FetchContent_Declare(
           doctest
           GIT_REPOSITORY https://github.com/doctest/doctest.git
           GIT_TAG v2.4.9
           GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(doctest)

file(
  GLOB_RECURSE TEST_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE header_lib static_lib shared_lib doctest_with_main)
target_include_directories(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
