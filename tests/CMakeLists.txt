# Fetching GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
#FetchContent_MakeAvailable(googletest)

# Executables
add_executable(heston_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests.cc)

# Linking
target_link_libraries(heston_tests 
    PRIVATE 
        hestonFFT_core
#        GTest::gtest_main
)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src")
target_include_directories(heston_tests PRIVATE ${SRC_DIR})
