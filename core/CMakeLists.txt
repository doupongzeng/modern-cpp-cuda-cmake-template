project(main CXX)

find_package(OpenCV 4 REQUIRED)

file(
    GLOB_RECURSE MAIN_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_executable(main ${MAIN_SOURCE_FILES})

# link libraries
target_link_libraries(main PRIVATE cuda_lib ${OpenCV_LIBRARIES})
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
