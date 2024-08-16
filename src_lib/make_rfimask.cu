#include <gputils/cuda_utils.hpp>
#include "../include/n2k/launch_rfimask_maker.hpp"

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Arguments:
//
//   uint4 rfimask[n128];       // i.e. number of bits is 128*n128
//   float sk_sigma[nbits/ds];  // where nbits = 128*n128
//   uint n128;                 // defines array dimensions
//   uint ds;                   // downsampling factor
//   float thresh;              // masking threshold (i.e. 5.0f for 5 sigma)
//
// Constraints (caller must check):
//
//   - n128 < 2^30.
//   - nbits (=128*n128) is a multiple of ds.
//   - ds is a multiple of 32.
//   - ds > 0.

__global__ void make_rfimask(uint4 *rfimask, const float *sk_sigma, uint n128, uint ds, float thresh)
{
    static constexpr uint ALL_ONES = 0xffffffffU;
	
    uint k = ds >> 5;
    uint t = (blockIdx.x * blockDim.x) + threadIdx.x;  // "global" thread id
    uint isrc = (t << 2) / k;    // one integer divide needed in this kernel
    uint imod = (t << 2) - k*isrc;
    uint4 rm;
    
    rm.x = (sk_sigma[isrc] < thresh) ? ALL_ONES : 0;
    isrc = (imod < (k-1)) ? (isrc) : (isrc+1);
    imod = (imod < (k-1)) ? (imod+1) : 0;

    rm.y = (sk_sigma[isrc] < thresh) ? ALL_ONES : 0;
    isrc = (imod < (k-1)) ? (isrc) : (isrc+1);
    imod = (imod < (k-1)) ? (imod+1) : 0;
    
    rm.z = (sk_sigma[isrc] < thresh) ? ALL_ONES : 0;
    isrc = (imod < (k-1)) ? (isrc) : (isrc+1);
    imod = (imod < (k-1)) ? (imod+1) : 0;
    
    rm.w = (sk_sigma[isrc] < thresh) ? ALL_ONES : 0;

    if (t < n128)
	rfimask[t] = rm;
}


// Arguments:
//
//   uint rfimask[nbits/32];
//   float sk_sigma[nbits/ds];
//   long nbits;                // defines arary dimensions
//   long ds;                   // downsampling factor
//   float thresh;              // masking threshold (i.e. 5.0f for 5 sigma)
//
// Constraints (checked here):
//
//   - nbits < 2^37 (currently assumed in GPU kernel).
//   - nbits is a multiple of 128.
//   - nbits is a multiple of ds.
//   - ds is a multiple of 32.

void launch_rfimask_maker(uint *rfimask, const float *sk_sigma, long nbits, long ds, float thresh, cudaStream_t stream)
{
    if (nbits <= 0)
	throw runtime_error("launch_rfimask_maker(): nbits must be > 0");
    if (ds <= 0)
	throw runtime_error("launch_rfimask_maker(): ds must be > 0");
    if ((ds & 31) != 0)
	throw runtime_error("launch_rfimask_maker(): ds must be a multiple of 32");
    if ((nbits % ds) != 0)
	throw runtime_error("launch_rfimask_maker(): nbits must be a multiple of ds");
    if ((nbits & 127) != 0)
	throw runtime_error("launch_rfimask_maker(): nbits must be a multiple of 128");
    if ((nbits >= (1L << 37)))
	throw runtime_error("launch_rfimask_maker(): nbits must be < 2^37");

    // Note: since (nbits < 2^37), nblocks is safely below the max allowed value (2^31-1).
    long nblocks = (nbits + 4095) >> 12;

    make_rfimask <<< nblocks, 128, 0, stream >>>
	((uint4 *) rfimask, sk_sigma, nbits >> 7, ds, thresh);

    CUDA_PEEK("make_rfimask kernel launch");
}


// Helper for gputils::Array<> version of launch_rfimask_maker().
template<typename T>
static void check_1d_array(const Array<T> &a, const char *name)
{
    if (a.ndim != 1)
	throw runtime_error("launch_rfimask_maker(): expected '" + string(name) + "' to be a 1-d array, got shape=" + a.shape_str());
    if (!a.is_fully_contiguous())
	throw runtime_error("launch_rfimask_maker(): array '" + string(name) + "' is not contiguous");
    if (!a.on_gpu())
	throw runtime_error("launch_rfimask_maker(): array '" + string(name) + "' is not on the GPU");
}


// Arguments:
//
//   uint rfimask[nbits/32];
//   float sk_sigma[nbits/ds];
//   long ds;                   // downsampling factor
//   float thresh;              // masking threshold (i.e. 5.0f for 5 sigma)

void launch_rfimask_maker(Array<uint> &rfimask, const Array<float> &sk_sigma, long ds, float thresh, cudaStream_t stream)
{
    check_1d_array(rfimask, "rfimask");
    check_1d_array(sk_sigma, "sk_sigma");

    if ((sk_sigma.size * ds) != (rfimask.size * 32)) {
	stringstream ss;
	ss << "launch_rfimask_maker: (rfimask.size, sk_sigma.size, ds) = ("
	   << rfimask.size << ", " << sk_sigma.size << ", " << ds
	   << ") are mutually inconsistent";
	throw runtime_error(ss.str());
    }

    long nbits = rfimask.size * 32;

    // Call bare-pointer version of launch_rfimask_maker(), which also does argument-checking.
    launch_rfimask_maker(rfimask.data, sk_sigma.data, nbits, ds, thresh, stream);
}


}   // namespace n2k

