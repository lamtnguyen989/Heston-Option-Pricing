cmake_minimum_required(VERSION 3.22)
project(hestonFFT LANGUAGES CXX)

# Setting C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finding dependecies
find_package(Kokkos REQUIRED)
add_subdirectory(tpls/kokkos-fft)   # KokkosFFT library

# Making "core" library
add_library(hestonFFT_core INTERFACE)
target_link_libraries(hestonFFT_core 
    INTERFACE 
        Kokkos::kokkos 
        KokkosFFT::fft
)
target_include_directories(hestonFFT_core 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Executables
set(SRC_DIR "src")
set(SRC_FILE ${SRC_DIR}/main.cc)
add_executable(hestonFFT ${SRC_FILE})
target_link_libraries(hestonFFT PRIVATE hestonFFT_core)

## Decide whether or not to build testing binaries via -DBUILD_TESTING
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()