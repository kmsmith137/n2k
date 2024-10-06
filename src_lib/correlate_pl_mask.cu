#include "../include/n2k/internals.hpp"
#include "../include/n2k/device_inlines.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/device_mma.hpp>

using namespace std;
using namespace gputils;

namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Array layouts.
//
// We use the m8n8k128 int1 tensor core MMA. The input fragments are 8-by-128 int1 matrices,
// represented by an 'int' register. The output fragment is an 8-by-8 int32 matrix, represented
// by int[2]. The tensor core register assignments are:
//
//   A_{ij}    b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4     t0 t1 t2 t3 t4 <-> j5 j6 i0 i1 i2
//   B_{jk}    b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4     t0 t1 t2 t3 t4 <-> j5 j6 k0 k1 k2
//   C_{ik}    r0 <-> k0                             t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2
//
// The input array layouts are:
//
//   uint1 pl_mask[T/64][F][S][64];    // where (F,S) may be downsampled by (2,4,8)
//   uint1 rfimask[F][T];


// -------------------------------------------------------------------------------------------------


// Reads a 16-by-128 submatrix, as two 8-by-128 fragments in tensor core ordering (see above).
// The global memory pointer 'pl_adjusted' should include threadblock/warp offsets to point
// at the appropriate 16-by-128 submatrix, plus the 'pl_lane_offset' (see below) so that
// each lane points to a distinct 64-bit part of the 16-by-128 submatrix.

__device__ inline void read_pl_16_128(int pl[2][1], const ulong *pl_adjusted)
{
    // pl_lane_offset() has been written so that dereferencing the pointer gives
    // the following register assigment:
    //
    // b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4    t0 t1 t2 t3 t4 <-> i3 j6 i0 i1 i2    r <-> j5
    
    int2 t = *((const int2 *) pl_adjusted);
    
    // Now we're one warp_transpose() away from the desired tensor core assignment:
    //
    // b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4    t0 t1 t2 t3 t4 <-> j5 j6 i0 i1 i2    r <-> i3

    warp_transpose(t.x, t.y, 1);
    pl[0][0] = t.x;
    pl[1][0] = t.y;
}

// The following per-lane offset is applied to the 'pl' global memory pointer,
// so that each lane points to a distinct 64-bit part of a 16-by-128 submatrix.
//
// The pl_lane_offset is constructed so that dereferencing the pointer gives
// the following register assignment:
//
// b0 b1 b2 b3 b4 <-> j0 j1 j2 j3 j4    t0 t1 t2 t3 t4 <-> i3 j6 i0 i1 i2    r <-> j5
//
// Reminder: offsets/strides are 64-bit, since pl is (const ulong *).
// (In particular, the 't64_stride' argument should be F*S.)

__device__ inline int pl_lane_offset(uint t64_stride)
{
    // t0 t1 t2 t3 t4 <-> i3 j6 i0 i1 i2
    
    int ilo = (threadIdx.x >> 2) & 0x7;
    int ihi = (threadIdx.x & 0x1) << 3;
    int j = (threadIdx.x & 0x2) ? t64_stride : 0;
    return ilo + ihi + j;
}


// -------------------------------------------------------------------------------------------------


// FIXME comment this
//   r <-> k0      t0 t1 t2 t3 t4 <-> k1 k2 i0 i1 i2
__device__ inline void write_v_8_8(int *v_out, int v[2])
{
    int2 t;
    t.x = v[0];
    t.y = v[1];

    int2 *p = (int2 *) (v_out);
    p[threadIdx.x & 31] = t;
}


// -------------------------------------------------------------------------------------------------
//
// Parallelization:
//
//   blockIdx.z, threadIdx.z <-> downsampled (output) time
//   blockIdx.y, threadIdx.y <-> frequency channel
//   threadIdx.x <-> sub-tile of S-by-S matrix
//     (blockIdx.x not currently used, but will be used when S > 1024 is implemented)


template<bool Debug>
__global__ void __launch_bounds__(128,8)
correlate_pl_kernel_S16(int *V_out, const ulong *pl_mask, int Tout, int F, uint N128)
{
    if constexpr (Debug) {
	assert(blockDim.x == 32);
	assert(blockDim.y * blockDim.z == 4);
	assert(gridDim.x == 1);
	assert(gridDim.z * blockDim.z >= Tout);
	assert(gridDim.y * blockDim.y >= F);
    }
    
    constexpr int S = 16;    
    const uint tout = blockIdx.z * blockDim.z + threadIdx.z;
    const uint f = blockIdx.y * blockDim.y + threadIdx.y;

    if ((tout >= Tout) || (f >= F))
	return;   // okay since this kernel never calls __syncthreads()

    // ulong pl_mask[T/64][F][S];
    const uint t128_stride = (2*F) * S;  // 64-bit (ulong) stride
    pl_mask += ulong(tout) * ulong(N128) * ulong(t128_stride);
    pl_mask += (f * S) + pl_lane_offset(F*S);
    
    int pl[2][1];
    int v[3][2];

    v[0][0] = v[0][1] = v[1][0] = v[1][1] = v[2][0] = v[2][1] = 0;
    
    while (N128 > 0) {
	// Read 16-by-128 PL-submatrix.
	read_pl_16_128(pl, pl_mask);

	// Accumulate 16-by-16 V-matrix.
	mma_b1_m8_n8_k128(v[0], pl[0], pl[0], v[0]);
	mma_b1_m8_n8_k128(v[1], pl[1], pl[0], v[1]);
	mma_b1_m8_n8_k128(v[2], pl[1], pl[1], v[2]);
	
	pl_mask += t128_stride;
	N128--;
    }
    
    // int V_out[Tout][F][3][8][8]
    V_out += ulong(tout) * ulong(192*F);
    V_out += (192 * f);

    // Write to global memory.
    write_v_8_8(V_out, v[0]);
    write_v_8_8(V_out + 64, v[1]);
    write_v_8_8(V_out + 128, v[2]);
}


// -------------------------------------------------------------------------------------------------


void launch_correlate_pl_kernel(int *V_out, const ulong *pl_mask, long T, long F, long S, long Nds, cudaStream_t stream, bool debug)
{
    // FIXME asserts -> exceptions
    assert(V_out != nullptr);
    assert(pl_mask != nullptr);
    assert(T > 0);
    assert(F > 0);
    assert(S == 16);  // for now
    assert(Nds > 0);
    assert((Nds % 128) == 0);
    assert((T % Nds) == 0);

    // FIXME 32-bit overflow checks
    
    long Tout = T / Nds;
    uint N128 = Nds >> 7;
    dim3 nthreads = {32, 2, 2};
    dim3 nblocks = { 1, uint(F+1)/2, uint(Tout+1)/2 };
    
    if (debug)
	correlate_pl_kernel_S16<true> <<< nblocks, nthreads, 0, stream >>> (V_out, pl_mask, Tout, F, N128);
    else
	correlate_pl_kernel_S16<false> <<< nblocks, nthreads, 0, stream >>> (V_out, pl_mask, Tout, F, N128);

    CUDA_PEEK("correlate_pl_kernel_S16");
}


void launch_correlate_pl_kernel(Array<int> &V_out, const Array<ulong> &pl_mask, long Nds, cudaStream_t stream, bool debug)
{
    // pl_mask shape = (T/64, F, S)
    // V_out shape = (T/Nds, F, ntiles, 8, 8)
    
    check_array(pl_mask, "launch_correlate_pl_kernel", "pl_mask", 3, true);  // contiguous=true
    check_array(V_out, "launch_correlate_pl_kernel", "V_out", 5, true);      // contiguous=true
    
    long T = 64 * pl_mask.shape[0];
    long F = pl_mask.shape[1];
    long S = pl_mask.shape[2];

    // FIXME asserts -> exceptions
    assert((S % 16) == 0);
    long ntiles = ((S/8) * ((S/8) + 1)) / 2;

    assert(Nds > 0);
    assert((T % Nds) == 0);
    assert(V_out.shape_equals({T/Nds,F,ntiles,8,8}));
    
    launch_correlate_pl_kernel(V_out.data, pl_mask.data, T, F, S, Nds, stream, debug);
}


}  // namespace n2k
