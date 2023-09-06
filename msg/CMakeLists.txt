project(msg_lib CXX)

# set(CMAKE_CXX_FLAGS
#   "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wcast-align -Wcast-qual -Wdelete-non-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wpointer-arith -Wfloat-equal -Winit-self -Wredundant-decls -Wconversion -Wsign-promo -Wswitch-default -Wswitch-enum -Wvariadic-macros -Wwrite-strings -Wno-unknown-pragmas"
# )

string(REPLACE " " ";" my_list "${CMAKE_CXX_FLAGS}")

set(MY_FLAGS "")

foreach(item ${my_list})
  if(${item} STREQUAL "-Wold-style-cast")
  else()
    # list(APPEND MY_FLAGS ${item})
    set(MY_FLAGS "${MY_FLAGS} ${item}")
    # list(JOIN ${item} " " )
  endif()
endforeach()
set(CMAKE_CXX_FLAGS ${MY_FLAGS})

file(
  GLOB_RECURSE SHARED_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS src/*.c*)
add_library(${PROJECT_NAME} STATIC ${SHARED_SOURCE_FILES})

# Boost
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_MULTITHREADED ON)
# find_package(Boost 1.65.0.0 REQUIRED)
# target_compile_definitions(
#   my_shared_lib PRIVATE $<$<NOT:$<CONFIG:Debug>>:BOOST_UBLAS_NDEBUG>
#   $<$<NOT:$<CONFIG:Debug>>:BOOST_DISABLE_ASSERTS>)
# target_include_directories(my_shared_lib SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
list(APPEND CMAKE_PREFIX_PATH "/home/hz/test/env_list/fastdds-dev/install")
find_package(fastcdr REQUIRED)
find_package(fastrtps REQUIRED)

target_link_libraries(${PROJECT_NAME} fastcdr fastrtps)

# header
target_include_directories(${PROJECT_NAME}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
