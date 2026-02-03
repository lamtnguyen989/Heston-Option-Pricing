cmake_minimum_required(VERSION 3.22)
project(hestonFFT LANGUAGES CXX)

# Finding Kokkos
find_package(Kokkos REQUIRED)

# Adding dependecies
add_subdirectory(tpls/kokkos-fft)   # KokkosFFT library

# Sources
set(SRC_DIR "src")
set(SRC_FILE ${SRC_DIR}/main.cc)

# Executables
add_executable(hestonFFT ${SRC_FILE})
set_target_properties(hestonFFT PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Linking with shared headers and directories
target_include_directories(hestonFFT
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hestonFFT 
    Kokkos::kokkos 
    KokkosFFT::fft
)