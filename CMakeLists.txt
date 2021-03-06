project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/particle_filter.cpp src/main.cpp)
set(test_sources src/test.cpp src/particle_filter.cpp)

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE src/lib src)


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 


add_executable(particle_filter ${sources})
add_executable(test ${test_sources})

target_link_libraries(particle_filter z ssl uv uWS)
target_link_libraries(test Catch)


