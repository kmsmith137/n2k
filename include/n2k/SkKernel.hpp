#ifndef _N2K_SK_KERNEL_HPP
#define _N2K_SK_KERNEL_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor
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
// This source file declares 'class SkKernel', a wrapper class for a CUDA kernel
// with the following input/output arrays:
//
//   - input: single-feed S-arrays (S_0, S_1, S_2) indexed by (time, freq, station).
//   - output: feed-averaged SK-statistic and associated (b,sigma).
//   - output (optional): single-feed SK-statistic and associated (b,sigma).
//   - output (optional): boolean RFI mask based on feed-averaged SK-statistic.
//
// Recall that the input array to the SK-kernel (single-feed S-array) has the
// following memory layout:
//
//     ulong S[T][F][3][S];    // length-3 axis is {S0,S1,S2}
//
// The output SK-arrays have the following memory layouts:
//
//     // length-3 axes are {SK,b,sigma}
//     float SK[T][F][3][S];   // Case 1: single-feed SK-statistic
//     float SK[T][F][3];      // Case 2:feed-averaged SK-statistic
//
// The boolean RFI mask has the following memory layout:
//
//     int1 rfimask[F][T*Nds];    // high-resolution time index (multiplied by Nds)
//
// Note that in the RFI mask, time is the fastest varying index (usually time is
// slowest varying). This may create complications. For example, suppose that the
// ring buffer length 'T_ringbuf' is longer than a single kernel launch 'T_kernel':
//
//     uint rfimask[F][T_ringbuf * Nds / 32];   // T_ringbuf, not T_kernel
//
// where we represent the ringbuf as uint[F][T*Nds/32], not int1[F][T*Nds], since
// this is the way it works in code. From the perspective of the SkKernel, the
// rfimask is now a discontiguous subarray of a larger array. This can be handled
// by using the 'rfimask_fstride' kernel argument (see below) to the 32-bit frequency
// stride (T_ringbuf * Nds / 32). (If the rfimask array were contiguous, then
// 'rfimask_fstride' would be (T_kernel * Nds / 32).)
//
// The SkKernel uses the "bad feed" mask when computing the feed-averaged SK-statistic
// and the boolean RFI mask. Logically, the bad feed mask is a boolean 1-d array of
// length S=(2*D). We represent the bad feed mask as a length-S uint8_t array, where a
// zero value means "feed is bad", and any nonzero 8-bit value means "feed is good".
//
// In the larger X-engine context, two SkKernels are used (see block diagram in
// the ``RFI statistics computed on GPU'' section of the overleaf). The first
// SkKernel runs at ~1 ms, and computes feed-averaged SK-statistic and RFI mask
// (no single-feed SK-statistic). The second SkKernel runs at ~30 ms, and computes
// single-feed and feed-averaged SK-statistics (no RFI mask).
//
// When processing multiple "frames" of data, you should create a persistent SkKernel
// instance and call launch() for each frame, rather than creating a new SkKernel for
// each frame. This is because The SkKernel constructor is less "lightweight" than you
// might expect. (It allocates a few-KB array on the GPU, copies data from CPU to GPU,
// and blocks until the copy is complete).
//
// Reminder: users of the SK-arrays (either single-feed SK or feed-averaged SK) should
// test for negative values of sigma. There are several reasons that an SK-array element
// can be invalid (masked), and this is indicated by setting sigma to a negative value.


struct SkKernel
{
    // High-level parameters for the SkKernel.
    // See overleaf for precise descriptions.
    // We might define kotekan yaml config parameters which are in one-to-one
    // correspondence with these parameters.
    
    struct Params {
	double sk_rfimask_sigmas = 0.0;             // RFI masking threshold in "sigmas" (only used if out_rfimask != NULL)
	double single_feed_min_good_frac = 0.0;     // For single-feed SK-statistic (threshold for validity)
	double feed_averaged_min_good_frac = 0.0;   // For feed-averaged SK-statistic (threshold for validity)
	double mu_min = 0.0;                        // For single-feed SK-statistic (threshold for validity)
	double mu_max = 0.0;                        // For single-feed SK-statistic (threshold for validity)
	long Nds = 0;                               // Downsampling factor used to construct S012 array (i.e. SK-kernel input array)
    };

    // As noted above, the SkKernel constructor allocates a few-KB array on the GPU,
    // copies data from CPU to GPU, and blocks until the copy is complete.
    //
    // Note: params are specified at construction, but also can be changed freely between calls to launch():
    //    sk_kernel->params.sk_rfimask_sigmas = 3.0;   // this sort of thing is okay at any time

    SkKernel(const Params &params, bool check_params=true);
    
    Params params;

    // Bare-pointer launch() interface.
    // Launches asynchronosly (i.e. does not synchronize stream or device after launching kernel.)
    
    void launch(
        float *out_sk_feed_averaged,          // Shape (T,F,3)
	float *out_sk_single_feed,            // Shape (T,F,3,S), can be NULL
	uint *out_rfimask,                    // Shape (F,T*Nds/32), can be NULL
	const ulong *in_S012,                 // Shape (T,F,3,S)
	const uint8_t *in_bf_mask,            // Length S (bad feed mask)
	long rfimask_fstride,                 // Only used if (out_rfimask != NULL). NOTE: uint32 stride, not bit stride!
	long T,                               // Number of downsampled times in S012 array
	long F,                               // Number of frequency channels
	long S,                               // Number of stations (= 2 * dishes)
	cudaStream_t stream = 0,
	bool check_params = true) const;
    
    // gputils::Array<> interface to launch().
    // Launches asynchronosly (i.e. does not synchronize stream or device after launching kernel.)

    void launch(
        gputils::Array<float> &out_sk_feed_averaged,   // Shape (T,F,3)
	gputils::Array<float> &out_sk_single_feed,     // Either empty array or shape (T,F,3,S)
	gputils::Array<uint> &out_rfimask,             // Either empty array or shape (F,T*Nds/32), need not be contiguous
	const gputils::Array<ulong> &in_S012,          // Shape (T,F,3,S)
	const gputils::Array<uint8_t> &in_bf_mask,     // Length S (bad feed bask)
	cudaStream_t stream = 0) const;

    // Used internally by launch() + constructor.
    // You shouldn't need to call this directly.
    static void check_params(const Params &params);

    // Interpolation table, copied to GPU memory by constructor.
    gputils::Array<float> bsigma_coeffs;
    int device = -1;
};


}  // namespace n2k

#endif // _N2K_SK_KERNEL_HPP
