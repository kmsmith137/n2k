#ifndef _N2K_INTERPOLATION_HPP
#define _N2K_INTERPOLATION_HPP

#include "device_inlines.hpp"  // bank_conflict_free_load(), roll_forward(), roll_backward()

namespace n2k {
#if 0
}  // editor auto-indent
#endif


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
#if 0
    sigma_coeffs += (threadIdx.x & 7);
    c0 = sigma_coeffs[8*i];
    c1 = sigma_coeffs[8*i+8];
    c2 = sigma_coeffs[8*i+16];
    c3 = sigma_coeffs[8*i+24];
#elif 0
    sigma_coeffs += (threadIdx.x & 7);
    c0 = sigma_coeffs[8 * ((i+3) & ~3)];
    c1 = sigma_coeffs[8 * (((i+2) & ~3) + 1)];
    c2 = sigma_coeffs[8 * (((i+1) & ~3) + 2)];
    c3 = sigma_coeffs[8 * (((i) & ~3) + 3)];
    roll_forward(i, c0, c1, c2, c3);
#elif 0
     int s = (threadIdx.x >> 3) & 3;
     sigma_coeffs += (threadIdx.x & 7);
     c0 = sigma_coeffs[8 * (((i+s+3) & ~3) - s)];
     c1 = sigma_coeffs[8 * (((i+s+2) & ~3) + 1-s)];
     c2 = sigma_coeffs[8 * (((i+s+1) & ~3) + 2-s)];
     c3 = sigma_coeffs[8 * (((i+s) & ~3) + 3-s)];
     roll_forward(i+s, c0, c1, c2, c3);
#elif 0
     int s = (threadIdx.x >> 3) & 3;
     sigma_coeffs += (threadIdx.x & 7);
     c0 = bank_conflict_free_load<Debug> (sigma_coeffs + 8 * (((i+s+3) & ~3) - s));
     c1 = bank_conflict_free_load<Debug> (sigma_coeffs + 8 * (((i+s+2) & ~3) + 1-s));
     c2 = bank_conflict_free_load<Debug> (sigma_coeffs + 8 * (((i+s+1) & ~3) + 2-s));
     c3 = bank_conflict_free_load<Debug> (sigma_coeffs + 8 * (((i+s) & ~3) + 3-s));
     roll_forward(i+s, c0, c1, c2, c3);
#elif 0
     int u = (threadIdx.x & 0x18);  // 8*s
     sigma_coeffs += (threadIdx.x & 7);
     c0 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+u+24) & ~31) - u);
     c1 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+u+16) & ~31) + 8-u);
     c2 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+u+8) & ~31) + 16-u);
     c3 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+u) & ~31) + 24-u);
     roll_forward(i+s, c0, c1, c2, c3);
#elif 0
     int l = (threadIdx.x & 31);
     c0 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+l+24) & ~31) + 7-l);
     c1 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+l+16) & ~31) + 15-l);
     c2 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+l+8) & ~31) + 23-l);
     c3 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i+l) & ~31) + 31-l);
     roll_forward(i + (threadIdx.x >> 3), c0, c1, c2, c3);
#else
    int t = threadIdx.x;
    c0 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 24) & ~31) - t + 7);
    c1 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 16) & ~31) - t + 15);
    c2 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 8) & ~31) - t + 23);
    c3 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t ) & ~31) - t + 31);
    roll_forward(i + (t >> 3), c0, c1, c2, c3);
#endif
}


}  // namespace n2k

#endif // _N2K_INTERPOLATION_HPP
