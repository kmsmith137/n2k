#ifndef _N2K_INTERPOLATION_HPP
#define _N2K_INTERPOLATION_HPP

#include "device_inlines.hpp"  // bank_conflict_free_load(), roll_forward(), roll_backward()

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// t = (-1,0,1,2) returns (y0,y1,y2,y3) respectively.
// This function is tested in src_bin/test-helper-functions.cu.

template<typename T>
__host__ __device__ inline T cubic_interpolate(T t, T y0, T y1, T y2, T y3)
{
    constexpr T one_half = T(1) / T(2);
    constexpr T one_third = T(1) / T(3);
    
    T d01 = (t) * (y1 - y0);
    T d12 = (t-1) * (y2 - y1);

    T c12 = (t) * (y2 - y1);
    T c23 = (t-1) * (y3 - y2);
    
    T d012 = one_half * (t-1) * (c12 - d01);
    T c123 = one_half * (t) * (c23 - d12);
    
    T c0123 = one_third * (t+1) * (c123 - d012);
    return c0123 + d012 + c12 + y1;
}


// load_sigma_coeffs(sigma_coeffs, i, c0, c1, c2 ,c3)
//
// 'sigma_coeffs' points to an array of shape (N,8).
// The length-8 axis is a spectator, i.e. all 8 values are equal.
//
// This function is equivalent to:
//   c0 = sigma_coeffs[8*i];
//   c1 = sigma_coeffs[8*i+8];
//   c2 = sigma_coeffs[8*i+16];
//   c3 = sigma_coeffs[8*i+24];
//
// but is guaranteed bank conflict free. This function is tested in
// src_bin/test-helper-functions.cu.


template<bool Debug = false>
__device__ inline void load_sigma_coeffs(const float *sigma_coeffs, int i, float &c0, float &c1, float &c2, float &c3)
{
    int t = threadIdx.x;
    c0 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 24) & ~31) - t + 7);
    c1 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 16) & ~31) - t + 15);
    c2 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 8) & ~31) - t + 23);
    c3 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t ) & ~31) - t + 31);
    roll_forward(i + (t >> 3), c0, c1, c2, c3);
}



// load_bias_coeffs(bias_coeffs, i, y, c0, c1, c2 ,c3)
//
// 'bias_coeffs' points to an array of shape (N,4,2).
// The length-2 axis is a spectator, i.e. all 2 values are equal.
//
// This function is equivalent to:
//   c0 = sum(bias_coeffs[8*i+2*j] * y^j for j in [0,1,2,3]);
//   c1 = sum(bias_coeffs[8*i+2*j+8] * y^j for j in [0,1,2,3]);
//   c2 = sum(bias_coeffs[8*i+2*j+16] * y^j for j in [0,1,2,3]);
//   c3 = sum(bias_coeffs[8*i+2*j+24] * y^j for j in [0,1,2,3]);
//
// but is guaranteed bank conflict free. This function is tested in
// src_bin/test-helper-functions.cu.


template<bool Debug = false>
__device__ inline float load_bias_inner(const float *coeffs, float y0, float y1, float y2, float y3)
{
    int t = threadIdx.x;
    float c0 = bank_conflict_free_load<Debug> (coeffs + ((t+6) & ~7) + 1-t);
    float c1 = bank_conflict_free_load<Debug> (coeffs + ((t+4) & ~7) + 3-t);
    float c2 = bank_conflict_free_load<Debug> (coeffs + ((t+2) & ~7) + 5-t);
    float c3 = bank_conflict_free_load<Debug> (coeffs + (t & ~7) + 7-t);
    return c0*y0 + c1*y1 + c2*y2 + c3*y3;
}


template<bool Debug = false>
__device__ inline void load_bias_coeffs(const float *bias_coeffs, int i, float y, float &c0, float &c1, float &c2, float &c3)
{
    float y0 = 1.0f;
    float y1 = y;
    float y2 = y*y;
    float y3 = y2*y;
    roll_backward(threadIdx.x >> 1, y0, y1, y2, y3);

    int s = (threadIdx.x >> 3) & 3;
    c0 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s+3) & ~3) - s), y0, y1, y2, y3);
    c1 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s+2) & ~3) + 1-s), y0, y1, y2, y3);
    c2 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s+1) & ~3) + 2-s), y0, y1, y2, y3);
    c3 = load_bias_inner<Debug> (bias_coeffs + 8 * (((i+s) & ~3) + 3-s), y0, y1, y2, y3);
    roll_forward(i+s, c0, c1, c2, c3);
}

    
}  // namespace n2k

#endif // _N2K_INTERPOLATION_HPP
