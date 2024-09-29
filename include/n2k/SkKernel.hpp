#ifndef _N2K_SK_KERNEL_HPP
#define _N2K_SK_KERNEL_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor
#endif


// SkKernel: wrapper class for GPU kernel which computes the SK-statistic and RFI mask
// from the S_0, S_1, S_2 statistics.
//
// When processing multiple "frames" of data, you should create a persistent SkKernel
// instance and call launch() for each frame, rather than creating a new SkKernel for
// each frame. This is because The SkKernel constructor is less "lightweight" than you
// might expect (it allocates a few-KB array on the GPU, copies data from CPU to GPU,
// and blocks until the copy is complete).


struct SkKernel
{
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
    // Note: params are specified at construction, but also can be changed freely between calls to launch().
    // That is, this sort of thing is okay:
    //    sk_kernel->params.sk_rfimask_sigmas = 3.0;

    SkKernel(const Params &params, bool check_params=true);
    
    Params params;

    // Bare-pointer launch() interface.
    // Launches asynchronosly (i.e. does not synchronize stream or device after launching kernel.)
    
    void launch(
        float *out_sk_feed_averaged,          // Shape (T,F,3)
	float *out_sk_single_feed,            // Shape (T,F,3,S), can be NULL
	uint *out_rfimask,                    // Shape (F,T*Nds/32), can be NULL
	const uint *in_S012,                  // Shape (T,F,3,S)
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
	const gputils::Array<uint> &in_S012,           // Shape (T,F,3,S)
	const gputils::Array<uint8_t> &in_bf_mask,     // Length S (bad feed bask)
	cudaStream_t stream = 0) const;

    // Used internally by launch() + constructor.
    // You shouldn't need to call this directly.
    static void check_params(const Params &params);

    // Interpolation table, copied to GPU memory by constructor.
    gputils::Array<float> bsigma_coeffs;
    int device = -1;
};


// These functions are only used for testing.
// Reminder: x=log(mu)=log(S1/S0), and y=(1/N)=(1/S0).

extern double interpolate_sk_bias(double x, double y);
extern double interpolate_sk_sigma(double x);


}  // namespace n2k

#endif // _N2K_SK_KERNEL_HPP
