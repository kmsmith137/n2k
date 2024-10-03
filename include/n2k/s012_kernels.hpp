#ifndef _N2K_S012_KERNELS_HPP
#define _N2K_S012_KERNELS_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// RFI kernels are split between two .hpp files:
//
//    s012_kernels.hpp: kernels which create and downsample S-arrays.
//    SkKernel.hpp: kernel which computes SK-statistics and boolean RFI mask.
//
// For a description of the X-engine RFI flagging logic, see the high-level
// software overleaf ("RFI statistics computed on GPU" section). The RFI code
// will be hard to understand unless you're familiar with this document!
// 
// This file (s012_kernels.hpp) declares four GPU kernels which create and downsample S-arrays:
//
//    launch_s0_kernel(): computes S0 from the packet loss mask.
//    launch_s012_kernel(): computes S1, S2 from the E-array.
//    launch_s012_time_downsample_kernel(): downsamples S0,S1,S2 along time-axis.
//    launch_s012_station_downsample_kernel(): downsamples S0,S1,S2 along station-axis.
//
// Recall (from the overleaf) the following properties of S-arrays:
//
//    - S-array layout is
//
//         ulong S[Tcoarse][F][3][2*D];   // case 1: single-feed S-array
//         ulong S[Tcoarse][F][3];        // case 2: feed-summed S-array
//
//      where the length-3 axis “packs” (S0, S1, S2) into a single array.
//
//    - The S-arrays contain cumulants of the E-array. Schematically:
//
//         S0 = sum (PL)
//         S1 = sum (PL * |E|^2)
//         S2 = sum (PL * |E|^4)
//
//    - The single-feed S-array (case 1 above) is created by calling
//      launch_s0_kernel() and launch_s012_kernel(), and can be downsampled
//      further in time by calling launch_s012_time_downsample_kernel().
//
//    - The feed-summed S-array (case 2 above) is created by calling
//      launch_s012_station_downsample_kernel() on the single-feed S-array.


// -------------------------------------------------------------------------------------------------
//
// Kernel 1/4: launch_s0_kernel().
//
// Computes S0 from packet loss mask, downsampling in time by specified factor 'Nds'.
//
// Constraints:
//   - Nds must be even (but note that the SK-kernel requires Nds to be a multiple of 32)
//   - S must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of 128 (required by packet loss array layout, see s0_kernel.cu)
//   - T must be a multiple of Nds.
//
// Note: the packet loss mask has a nontrivial array layout. See either the software overleaf
// (section "Data sent from CPU to GPU in each kotekan frame"), or comments in s0_kernel.cu.
//
// A note on 'out_fstride' argument to launch_s0_kernel(): In the larger X-engine context,
// the s0_kernel is used to initialize "one-third" of the S012 array:
//
//   S[Tcoarse][F][3][2*D];   // s0_kernel initializes S[:,:,0,:]
//
// Therefore, from the perspective of the s0_kernel, the output array is discontiguous:
// the "stride" of the frequency axis is (6*D) instead of (2*D). This is implemented by
// passing out_fstride=6*D to launch_s0_kernel().


// Version 1: bare-pointer interface.
extern void launch_s0_kernel(
    ulong *S0,                   // output array, shape (T/Nds, F, S)
    const ulong *pl_mask,        // input array, shape (T/128, (F+3)/4, S/8), see above
    long T,                      // number of time samples in input array (before downsampling)
    long F,                      // number of frequency channels
    long S,                      // number of stations (= 2*D, where D is number of dishes)
    long Nds,                    // time downsampling factor
    long out_fstride,            // frequency stride in 'S0' array, see comment above.
    cudaStream_t stream=0);


// Version 2: gputils::Array<> interface.
// Note that there is no 'out_fstride' arugment, since 'S0' is a gputils::Array, which contains strides.

extern void launch_s0_kernel(
     gputils::Array<ulong> &S0,              // output array, shape (T/Nds, F, S)
     const gputils::Array<ulong> &pl_mask,   // input array, shape (T/128, (F+3)/4, S/8), see above
     long Nds,                               // time downsampling factor
     cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 2/4: launch_s12_kernel().
//
// Computes S1,S2 from E-array, downsampling in time by specified factor 'Nds'.
//
// Note: the s12_kernel does not need the packet loss mask as an input. This is because
// the E-array is assumed to be zeroed for missing packets. (See overleaf notes, section
// "Data sent from CPU to GPU in each kotekan frame".)
//
// A note on 'out_fstride' argument to launch_s12_kernel(): In the larger X-engine context,
// the s12_kernel is used to initialize "two-thirds" of the S012 array:
//
//   S[Tcoarse][F][3][2*D];   // s12_kernel initializes S[:,:,1:3,:]
//
// Therefore, from the perspective of the s12_kernel, the output array is discontiguous:
// the "stride" of the frequency axis is (6*D) instead of (4*D). This is implemented by
// passing out_fstride=6*D to launch_s12_kernel(). (You'll also add the offset (2*D) to
// the S-array base pointer, in order to write to S[:,:,1:3,:] rather than S[:,:,0:2,:].)


// Version 1: bare-pointer interface.
extern void launch_s12_kernel(
    ulong *S12,           // output array, shape (T/Nds, F, 2, S)
    const uint8_t *E,     // input int4+4 array, shape (T, F, S)
    long T,               // number of time samples in input array (before downsampling)
    long F,               // number of frequency channels
    long S,               // number of stations (= 2*D, where D is number of dishes)
    long Nds,             // time downsampling factor
    long out_fstride,     // frequency stride in 'S12' array, see comment above.
    cudaStream_t stream=0);


// Version 2: gputils::Array<> interface.
// Note that there is no 'out_fstride' arugment, since 'S12' is a gputils::Array, which contains strides.

extern void launch_s12_kernel(
    gputils::Array<ulong> &S12,         // output array, shape (T/Nds, F, 2, S)
    const gputils::Array<uint8_t> &E,   // input int4+4 array, shape (T, F, S)
    long Nds,                           // time downsampling factor
    cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 3/4: launch_s012_time_downsample_kernel().
//
// Downsamples S012 along time axis, i.e. input/output arrays are
//
//     Sin[T][F][3][S];        // input array
//     Sout[T/Nds][F][3][S];   // output array
//
// Context: In the overleaf notes, "RFI statistics computed on GPU" section, you'll see
// a block diagram with an arrow labelled "Downsample in time", where the S012 arrays are
// downsampled from ~1 ms to ~30 ms. This kernel implements this step.
//
// Note: the last three indices in the above arrays are "spectator" indices as far as
// this kernel is concerned, and can be replaced by a single spectator index with length
// M = (3*F*S). (As usual, S denotes number of stations, i.e. 2*D, where D = number of dishes.)


// Version 1: bare-pointer interface.
extern void launch_s012_time_downsample_kernel(
    ulong *Sout,        // output array, shape (T/Nds,3,F,S) or equivalently (T/Nds,M)
    const ulong *Sin,   // input array, shape (T,3,F,S) or equivalently (T,M)
    long T,             // number of time samples before downsampling
    long M,             // number of spectator indices (3*F*S), see above
    long Nds,           // time downsampling factor
    cudaStream_t stream=0);

// Version 2: gputils::Array<> interface.
extern void launch_s012_time_downsample_kernel(
    gputils::Array<ulong> &Sout,        // output array, shape (T/Nds,3,F,S)
    const gputils::Array<ulong> &Sin,   // input array, shape (T,3,F,S)
    long Nds,                           // time downsampling factor
    cudaStream_t stream=0);


// -------------------------------------------------------------------------------------------------
//
// Kernel 4/4: launch_s012_station_downsample_kernel().
//
// Downsamples S012 along station axis, i.e. input/output arrays are
//
//     Sin[T][F][3][S];     // input array
//     Sout[T][F][3];       // output array
//
// Context: In the overleaf notes, "RFI statistics computed on GPU" section, you'll see
// a block diagram with an arrow labelled "Sum S0, S1, S2 over feeds", where the feed-summed
// S-array is comptued from the single-feed S-array. This kernel implements this step.
//
// Note: this kernel uses the "bad feed" mask. Logically, this is a boolean 1-d array of
// length S=(2*D). We represent the bad feed mask as a length-S uint8_t array, where a
// zero value means "feed is bad", and any nonzero 8-bit value means "feed is good".
//
// Note: the first three indices in the above arrays are "spectator" indices as far as
// this kernel is concerned, and can be replaced by a single spectator index with length
// M = (3*T*F).


// Version 1: bare-pointer interface.
extern void launch_s012_station_downsample_kernel(
    ulong *Sout,             // output array, shape (T,F,3) or equivalently (M,)
    const ulong *Sin,        // input array, shape (T,F,3,S) or equivalently (M,S)
    const uint8_t *bf_mask,  // bad feed mask, shape (S,), see above
    long M,                  // number of spectator indices (3*T*F), see above
    long S,                  // number of stations
    cudaStream_t stream=0);

// Version 2: gputils::Array<> interface.
extern void launch_s012_station_downsample_kernel(
    gputils::Array<ulong> &Sout,              // output array, shape (T,F,3)
    const gputils::Array<ulong> &Sin,         // input array, shape (T,F,3,S)
    const gputils::Array<uint8_t> &bf_mask,   // bad feed mask, shape (S,), see above
    cudaStream_t stream=0);


} // namespace n2k

#endif // _N2K_S012_KERNELS_HPP
