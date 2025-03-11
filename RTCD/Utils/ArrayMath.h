#pragma once

#include <algorithm>
#include <array>
#include <math.h>


template <size_t N>
std::array<float, N> operator-(const std::array<float, N>& a, const std::array<float, N>& b) {
    std::array<float, N> c;
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::minus<float>());
    return c;
}

template <size_t N>
std::array<float, N> operator+(const std::array<float, N>& a, const std::array<float, N>& b) {
    std::array<float, N> c;
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<float>());
    return c;
}

template <size_t N>
std::array<float, N> operator*(const std::array<float, N>& a, float b) {
    std::array<float, N> c;
    std::transform(a.begin(), a.end(), c.begin(), [b](float x) { return x * b; });
    return c;
}

template <size_t N>
std::array<float, N> operator/(const std::array<float, N>& a, float b) {
    std::array<float, N> c;
    std::transform(a.begin(), a.end(), c.begin(), [b](float x) { return x / b; });
    return c;
}

template <size_t N>
std::array<float, N> operator*(float b, const std::array<float, N>& a) {
    std::array<float, N> c;
    std::transform(a.begin(), a.end(), c.begin(), [b](float x) { return x * b; });
    return c;
}

template <size_t N>
std::array<float, N> operator/(float b, const std::array<float, N>& a) {
    std::array<float, N> c;
    std::transform(a.begin(), a.end(), c.begin(), [b](float x) { return x / b; });
    return c;
}


template <size_t N>
float norm(const std::array<float, N>& a) {
    float sum = 0;
    for (float x : a) {
        sum += x * x;
    }
    return sqrt(sum);
}