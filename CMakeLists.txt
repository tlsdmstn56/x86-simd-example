cmake_minimum_required(VERSION 3.0)

project(x86-simd-example)

add_executable(simd-example src/main.cpp)

target_compile_options(simd-example 
    PRIVATE "-march=sandybridge"
)

target_compile_features(simd-example 
    PRIVATE cxx_std_11
)