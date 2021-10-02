# CMakeList.txt : CMake project for SweeterThanCady, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.

include_directories(C:/Users/dest/sweeterthancandy/sweeterthancady/include)
include_directories(C:/work/boost_1_77_0/boost_1_77_0)

if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    if (_cpp_latest_flag_supported)
        add_compile_options("/std:c++latest")
    endif()
endif()

add_executable (SweeterThanCady "SweeterThanCady.cpp" "SweeterThanCady.h")

# TODO: Add tests and install targets if needed.
