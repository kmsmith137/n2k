#include "../include/n2k/pl_kernels.hpp"
#include "../include/n2k/internals/internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Input: 32-bit integer, represented as bit sequence [ x0 x1 ... x32 ].
//
// Output: "Doubled" bit sequence, represented as two 32-bit integers:
//   ret.x = [ x0 x0 x1 x1 ... x15 x15 ]
//   ret.y = [ x16 x16 x17 x17 ... x31 x31 ]
//
// Note: I tried a fancy bit-transposing implementation with lop3() and
// __byte_perm(), but this boneheaded implementation was faster (?!).

__device__ inline uint2 double_bits(uint x)
{
    uint2 ret{0U,0U};

    for (uint i = 0; i < 16; i++) {
	uint x_shifted = x >> i;
	uint bit_pair = 3U << (2*i);
	
	if (x_shifted & 1U)
	    ret.x |= bit_pair;
	if (x_shifted & 0x10000U)
	    ret.y |= bit_pair;
    }

    return ret;
}


// Kernel args:
//   F = (number of output freqs)
//   M = (number of input times) / 64
//   N = (number of stations) * 2
//   pl_out = uint array of shape (2*M, F, N)
//   pl_in = uint array of shape (M, (F+3)/4, N)

__global__ void pl_mask_expand_kernel(uint *pl_out, const uint *pl_in, int F, int M, int N)
{
    const long Fin = (F+3) >> 2;
    
    // Parallelization: x <-> n, y <-> f, z <-> m
    long n = (blockIdx.x * blockDim.x) + threadIdx.x;
    long f = (blockIdx.y * blockDim.y) + threadIdx.y; 
    long m = (blockIdx.z * blockDim.z) + threadIdx.z;
    
    bool valid = (f < Fin) && (m < M) && (n < N);
    long nf_out = valid ? min(F-4*f,4L) : 0;

    // Ensure array accesses are within bounds.
    f = (f < Fin) ? f : (Fin-1);
    m = (m < M) ? m : (M-1);
    n = (n < N) ? n : (N-1);
    
    // pl_in = uint array of shape (M, Fin, N)
    // After these shifts, 'pl_in' points to a scalar.
    
    pl_in += m*Fin*N;  // 64-bit safe
    pl_in += f*N + n;  // 64-bit safe

    // pl_out = uint array of shape (2*M, F, N)
    // After these shifts, 'pl_out' points to an array of shape (nf_out, 2) with strides (N, 1).

    long mout = 2*m + (n & 1);
    long nout = n & ~1;
    pl_out += mout * long(F) * long(N);  // 64-bit safe
    pl_out += (4*f*N + nout);            // 64-bit safe

    // Read input mask.
    uint x = *pl_in;
    uint2 y = double_bits(x);
	
    // Write (expanded) output mask.
    for (long i = 0; i < nf_out; i++)
	*((uint2 *) (pl_out + i*N)) = y;
}


void launch_pl_mask_expander(ulong *pl_out, const ulong *pl_in, long Tout, long Fout, long Sds, cudaStream_t stream)
{
    if (!pl_out)
	throw runtime_error("launch_pl_mask_expander: 'pl_out' must be non-NULL");
    if (!pl_in)
	throw runtime_error("launch_pl_mask_expander: 'pl_in' must be non-NULL");
    if (Tout <= 0)
	throw runtime_error("launch_pl_mask_expander: expected Tout > 0");
    if (Fout <= 0)
	throw runtime_error("launch_pl_mask_expander: expected Fout > 0");
    if (Sds <= 0)
	throw runtime_error("launch_pl_mask_expander: expected Sds > 0");
    if (Tout & 127)
	throw runtime_error("launch_pl_mask_expander: expected Tout to be a multiple of 128");
    if (Sds & 15)
	throw runtime_error("launch_pl_mask_expander: expected Sds to be a multiple of 16");

    long M = Tout / 128;
    long N = Sds * 2;
    
    if ((N > INT_MAX) || (Fout > INT_MAX) || (M > INT_MAX))
	throw runtime_error("launch_pl_mask_expander: 32-bit overflow");
    
    dim3 nblocks, nthreads;
    gputils::assign_kernel_dims(nblocks, nthreads, N, Fout, M);  // x <-> n, y <-> f, z <-> m

    pl_mask_expand_kernel <<< nblocks, nthreads, 0, stream >>>
	((uint *) pl_out, (const uint *) pl_in, Fout, M, N);

    CUDA_PEEK("pl_mask_expand_kernel");
}


void launch_pl_mask_expander(Array<ulong> &pl_out, const Array<ulong> &pl_in, cudaStream_t stream)
{
    check_array(pl_out, "launch_pl_mask_expander", "pl_out", 3, true);  // contiguous=true
    check_array(pl_in, "launch_pl_mask_expander", "pl_in", 3, true);    // contiguous=true

    long Tout = 64 * pl_out.shape[0];
    long Fout = pl_out.shape[1];
    long Sds = pl_out.shape[2];

    if (!pl_in.shape_equals({Tout/128, (Fout+3)/4, Sds})) {
	stringstream ss;
	ss << "launch_pl_mask_expander: pl_out.shape=" << pl_out.shape_str()
	   << " and pl_in.shape=" << pl_in.shape_str()
	   << " are inconsistent (expected pl_in.shape=("
	   << (Tout/128) << "," << ((Fout+3)/4) << "," << Sds 
	   << "))";
	throw runtime_error(ss.str());
    }

    launch_pl_mask_expander(pl_out.data, pl_in.data, Tout, Fout, Sds, stream);
}


}  // namespace n2k
