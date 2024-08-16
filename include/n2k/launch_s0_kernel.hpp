#ifndef _N2K_HPP
#define _N2K_HPP

#include <gputils/Array.hpp>


namespace n2k {
#if 0
}  // editor auto-indent
#endif

// launch_s0_kernel()
//
// Arguments:
//
//   uint s0[T/ds][F][S];                  // output array, (downsampled time index, freq channel, station)
//   ulong pl_mask[T/128][(F+3)/4][S/8];   // input array, packet loss mask
//   long T;                               // number of time samples
//   long F;                               // number of freq channels
//   long S;                               // number of stations (= dish+pol pairs)
//   long ds;                              // time downsampling factor
//
// Constraints:
//
//   - ds must be even (but note that the make_rfimask kernel requires ds to be a multiple of 32)
//   - S must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of ds.
//
// Note: see s0_kernel.cu for description and discussion of packet loss mask array layout.


// Bare pointer interface.
extern void launch_s0_kernel(uint *s0, const ulong *pl_mask, long T, long F, long S, long ds, cudaStream_t stream=0);

// gputils::Array<> interface.
extern void launch_s0_kernel(gputils::Array<uint> &s0, const gputils::Array<ulong> &pl_mask, long ds, cudaStream_t stream=0);


} // namespace n2k

#endif // _N2K_HPP
