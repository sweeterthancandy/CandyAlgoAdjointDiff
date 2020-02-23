
cmake_minimum_required(VERSION 3.7)

project(Cady)


set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_FLAGS "-std=c++14")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-braces")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

find_package(Boost  REQUIRED filesystem program_options system log timer COMPONENTS)
add_definitions(-DBOOST_LOG_DYN_LINK)

add_definitions(-Dint_p_NULL=0)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)


add_executable( driver driver.cpp )
target_link_libraries( driver pthread ${Boost_LIBRARIES})

