#ifndef _N2K_LAUNCH_RFIMASK_MAKER_HPP
#define _N2K_LAUNCH_RFIMASK_MAKER_HPP

#include <gputils/Array.hpp>


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// launch_rfimask_maker()
//
// Arguments:
//
//    uint rfimask[nbits/32];      // output array
//    float sk_sigma[nbits/ds];    // input array
//    long nbits;                  // defines dimensions of input/output arrays
//    long ds;                     // downsampling factor, also defines dimensions
//    float thresh;                // masking threshold in 'sk_sigma' array (e.g. 5.0 for 5 sigma)
//
// Constraints:
//
//   - nbits must be a multiple of 128, and a multiple of ds.
//   - ds must be a multiple of 32.
//
// Note: in this kernel, we treat 'rfimask' and 'sk_sigma' as 1-d arrays, even though
// in the larger correlator, they are 2-d arrays:
//
//    uint rfimask[F][T/32];
//    float sk_sigma[F][T/ds];
//
// This is because the frequency index is a "spectator" index, so we can flatten these arrays and
// treat them as 1-d arrays with nbits=F*T.
//
// These functions are defined in make_rfimask.cu.


// Bare pointer interface.
extern void launch_rfimask_maker(uint *rfimask, const float *sk_sigma, long nbits, long ds, float thresh, cudaStream_t stream=0);

// gputils::Array<> interface.
extern void launch_rfimask_maker(gputils::Array<uint> &rfimask, const gputils::Array<float> &sk_sigma, long ds, float thresh, cudaStream_t stream=0);


}  // namespace n2k

#endif // _N2K_HPP
