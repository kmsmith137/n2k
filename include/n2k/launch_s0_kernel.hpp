#ifndef _N2K_LAUNCH_S0_KERNEL_HPP
#define _N2K_LAUNCH_S0_KERNEL_HPP

#include <gputils/Array.hpp>


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------

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

// FIXME rename ds -> Tds
extern void launch_s0_kernel(uint *s0, const ulong *pl_mask, long T, long F, long S, long ds, cudaStream_t stream=0);
extern void launch_s0_kernel(gputils::Array<uint> &s0, const gputils::Array<ulong> &pl_mask, long ds, cudaStream_t stream=0);

extern void launch_s12_kernel(uint *S12, const uint8_t *E, long Nds, long Tout, long F, long S, long out_fstride, cudaStream_t stream=0);
extern void launch_s12_kernel(gputils::Array<uint> &S12, const gputils::Array<uint8_t> &E, long Nds, cudaStream_t stream=0);

extern void launch_s012_time_downsample_kernel(uint *Sout, const uint *Sin, long Nds, long Tout, long M, cudaStream_t stream=0);
extern void launch_s012_time_downsample_kernel(gputils::Array<uint> &Sout, const gputils::Array<uint> &Sin, long Nds, cudaStream_t stream=0);

// M is number of spectator indices (3*T*F)
extern void launch_s012_station_downsample_kernel(uint *Sout, const uint *Sin, const uint8_t *bf_mask, long M, long S, cudaStream_t stream=0);
extern void launch_s012_station_downsample_kernel(gputils::Array<uint> &Sout, const gputils::Array<uint> &Sin, const gputils::Array<uint8_t> &bf_mask, cudaStream_t stream=0);

			
} // namespace n2k

#endif // _N2K_LAUNCH_S0_KERNEL_HPP
