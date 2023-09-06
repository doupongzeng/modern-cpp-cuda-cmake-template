project(main CXX CUDA)

file(
    GLOB_RECURSE MAIN_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c* msg/*.c*)

add_executable(pub pub.cpp ${MAIN_SOURCE_FILES})
add_executable(sub sub.cpp ${MAIN_SOURCE_FILES})

list(APPEND CMAKE_PREFIX_PATH "/home/hz/test/env_list/fastdds-dev/install")

find_package(fastcdr REQUIRED)
find_package(fastrtps REQUIRED)

# link libraries
target_link_libraries(pub PRIVATE fastrtps fastcdr msg_lib)
target_link_libraries(sub PRIVATE fastrtps fastcdr msg_lib)

target_include_directories(pub PUBLIC include)
target_include_directories(sub PUBLIC include)
