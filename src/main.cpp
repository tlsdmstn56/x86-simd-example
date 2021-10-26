#include <iostream>
#include <string>
#include <chrono>
#include <xmmintrin.h>
#include <immintrin.h>
#include <iomanip>
#include <random>

void MultNative(int N, float v, float *arr)
{
    for (int i = 0; i < N; ++i)
    {
        arr[i] *= v;
    }
}

void MultSSE(int N, float v, float *arr)
{
    __m128 m2 = _mm_set_ps(v, v, v, v);
    float *end = arr + N;
    for (; arr < end; arr += 4)
    {
        __m128 m1 = _mm_load_ps(arr);
        __m128 sum = _mm_mul_ps(m1, m2);
        _mm_store_ps(arr, sum);
    }
}

void MultAVX(int N, float v, float *arr)
{
    __m256 m2 = _mm256_set_ps(v, v, v, v, v, v, v, v);
    float *end = arr + N;
    for (; arr < end; arr += 8)
    {
        __m256 m1 = _mm256_load_ps(arr);
        __m256 sum = _mm256_mul_ps(m1, m2);
        _mm256_store_ps(arr, sum);
    }
}

void RunBenchmark(void (*f)(int, float, float *), int N, float v, float *arr, const char *name)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t1 = Clock::now();
    f(N, v, arr);
    auto t2 = Clock::now();
    std::cout << std::setw(10) << name << ": ";
    std::cout << (t2 - t1).count() << " ns\n";
}

float RandFloat()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dist(0, 1000);
    return dist(rng);
}

void ResetArray(float* arr, int size, float v)
{
    std::fill(arr, arr+size, v);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Error: too few or too many arguments are provided\n";
        std::cout << "Syntax: " << argv[0] << " <N> \n";
        return 1;
    }
    int N = std::stoi(argv[1]);
    float arrInitialValue = RandFloat();
    float multOperand = RandFloat();
    float *arr = (float *)_mm_malloc(N * sizeof(float), 32);

    MultNative(N, multOperand, arr);

    ResetArray(arr, N, arrInitialValue);
    RunBenchmark(MultNative, N, multOperand, arr, "Naive");

    ResetArray(arr, N, arrInitialValue);
    RunBenchmark(MultSSE, N, multOperand, arr, "SSE");

    ResetArray(arr, N, arrInitialValue);
    RunBenchmark(MultAVX, N, multOperand, arr, "AVX");

    return 0;
}