#include <gputils/cuda_utils.hpp>
#include "../include/n2k/launch_s0_kernel.hpp"

using namespace std;
using namespace gputils;


// This is a good place to describe the Array layout for PL mask:
//
//   - The PL mask is logically an array bool[T/2][F/4][S/8], where
//     T is the number of time samples, F is number of freq channels,
//     and S is the number of "stations" (i.e. dish+pol pairs).
//     The array layout is non-trivial and described as follows.
//
//   - Separate time index into indices (t128, t2, t1):
//        t = 128*t128 + 2*t2 + t1    (where 0 <= t2 < 64 and 0 <= t1 < 2)
//
//   - Separate freq index into indices (f4, f1)
//        f = 4*t4 + f1               (where 0 <= f1 < 4)
//
//   - Separate station index into indices (s8, s1)
//        s = 8*s8 + s1               (where 0 <= s1 < 8)
//
//   - Then the PL mask uses this array layout:
//
//        // Usage: mask[t128][f4][s8][t2]
//        int1 mask[T/128][F/4][S/8][64]
//
//      where indices are ordered "C-style" from slowest to fastest varying,
//      and little-endian bit ordering is assumed when interpreting int1[...].
//
//    - An equivalent layout which we usually use in code:
//
//        // Usage: mask[t128][f4][s8] & (1UL << t2)
//        uint64 mask[T/128][F/4][S/8]
//
//    - The total memory footprint of the PL mask is 512 times smaller than
//      the corresponding electric field array. E.g. in CHORD, the electric
//      field bandwidth is 10 GB/s/GPU, and the PL mask bandwidth is
//      20 MB/s/GPU.
//
//    - This layout assumes that T is a multiple of 128. I assume this is okay,
//      since we can choose the kotekan frame size to be a multiple of 128.
//
//    - In order for this layout to have good cache alignment properties, we
//      assume that the number of stations S is a multiple of 128. This is the case
//      for all of our current projects (CHIME, HIRAX, CHORD, CHORD pathfinder).
//
//    - If the number of frequency channels F is not a multiple of 4, we can
//      generalize the array layouts to:
//
//        // Usage: mask[t128][f//4][s8][t2]
//        int1 mask[T/128][(F+3)//4][S/8][64]
//
//      (I don't know whether this generalization will be useful, but it doesn't
//       seem to be extra work to implement it in the GPU kernels, so I'm planning
//       to implement it.)


namespace n2k {
#if 0
}  // editor auto-indent
#endif


__device__ uint _cmask(int b)
{
    b = (b >= 0) ? b : 0;
    return (b < 32) ? (1U << b) : 0;
}


// s0_kernel() arguments:
//
//   uint4 s0[T/ds][F][S/4];                  // output array, (downsampled time index, freq channel, station)
//   uint pl_mask[T/128][(F+3)/4][S/8][2];    // input array, packet loss mask
//   long T;                                  // number of time samples
//   long F;                                  // number of freq channels
//   long S;                                  // number of stations (= dish+pol pairs)
//   long ds;                                 // time downsampling factor
//
// Constraints (checked in launch_s0_kernel() below)
//
//   - ds must be even.
//   - S must be a multiple of 128.
//   - T must be a multiple of 128.
//   - T must be a multiple of ds.
//
// Notes on parallelization:
//
//   - Each warp independently processes one tds index, 4 freqs, and 128 stations.
//     Each thread processes one (t/32) index, 4 freqs, and 8 stations.
//
//   - Within the warp, the thread mapping is
//       t4 t3 t2 t1 t0 <-> (s/64) (s/32) (s/16) (s/8) (time/32)
//
//   - Within the larger kernel, the warp mapping is:
//       wz wy wx <-> (tds) (f/4) (s/128)
//
// FIXME think carefully about int32 overflows!!

__global__ void s0_kernel(uint4 *s0, const uint *pl, int T, int F, int S, int ds)
{
    static constexpr uint ALL_LANES = 0xffffffffU;
    
    // Warp location within larger kerenl
    int tds = (blockIdx.z * blockDim.z + threadIdx.z);              // output time
    int fds = (blockIdx.y * blockDim.y + threadIdx.y);              // input (f/4)
    int sds = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) << 4;  // input (s/8), laneId not included, multiple of 16

    int Fds = (F+3) >> 2;      // number of downsampled freq channels in 'pl_mask'.
    int nf = min(4, F-4*fds);  // number of frequency channels to write

    // These tests guarante that we don't write past the edge of memory.
    
    if (tds*ds >= T)
	return;
    if (4*fds >= F)
	return;
    if (8*sds >= S)
	return;

    // Shift input pointer. Note that no time shift is not applied, but laneId is applied.
    // Before the shifts, 'pl' has shape uint[T/128, Fds, S/8, 2].
    // After these shifts, 'pl' has shape uint[T/128] and stride Fds * (S/4).
    
    pl += (sds << 1);
    pl += fds * (S >> 2);
    pl += (threadIdx.x & 31);  // laneId
    long pl_stride = long(Fds) * long(S >> 2);

    // Shift output pointer, including time and laneId.
    // Before the shifts, 's0' has shape uint4[T/ds, F, S/4].
    // After the shifts, 's0' has shape uint4[4] and stride (S/4).
    
    s0 += (sds << 1);
    s0 += (fds * S);           // (fds << 2) * (S >> 2)
    s0 += tds * F * (S >> 2);
    s0 += (threadIdx.x & 31);  // laneId
    int s0_stride = (S >> 2);
    
    // [t2_lo:t2_hi) = range of t2 values processed on this warp.
    int t2_lo = tds * (ds >> 1);
    int t2_hi = t2_lo + (ds >> 1);
    
    // [t128_lo:128_hi) = range of t128 values processed on this warp.
    int t128_lo = (t2_lo >> 6);
    int t128_hi = ((t2_hi-1) >> 6) + 1;

    uint s0_accum = 0;
    
    for (int t128 = t128_lo; t128 < t128_hi; t128++) {
	uint x = pl[t128 * pl_stride];
	int t2 = (t128 << 6) + ((threadIdx.x & 1) << 5);
	uint mask = _cmask(t2_hi - t2) - _cmask(t2_lo - t2);
	s0_accum += __popc(x & mask);
    }

    s0_accum <<= 1;
    s0_accum += __shfl_sync(ALL_LANES, s0_accum, threadIdx.x ^ 0x1);

    uint4 s0_x4;
    s0_x4.x = s0_accum;
    s0_x4.y = s0_accum;
    s0_x4.z = s0_accum;
    s0_x4.w = s0_accum;    

    for (int f = 0; f < nf; f++)
	s0[f * s0_stride] = s0_x4;
}


// launch_s0_kernel() arguments, bare pointer version:
//
//   uint s0[T/ds][F][S];                  // output array, (downsampled time index, freq channel, station)
//   ulong pl_mask[T/128][(F+3)/4][S/8];   // input array, packet loss mask
//   long T;                               // number of time samples
//   long F;                               // number of freq channels
//   long S;                               // number of stations (= dish+pol pairs)
//   long ds;                              // time downsampling factor
//
// Constraints (checked here)
//
//   - ds must be even.
//   - S must be a multiple of 128.
//   - T must be a multiple of 128.
//   - T must be a multiple of ds.
//
// Notes on parallelization:
//
//   - Each warp independently processes one tds index, 4 freqs, and 128 stations.
//     Each thread processes one (t/32) index, 4 freqs, and 8 stations.
//
//   - Within the warp, the thread mapping is
//       t4 t3 t2 t1 t0 <-> (s/64) (s/32) (s/16) (s/8) (time/32)
//
//   - Within the larger kernel, the warp mapping is:
//       wz wy wx <-> (tds) (f/4) (s/128)

void launch_s0_kernel(uint *s0, const ulong *pl_mask, long T, long F, long S, long ds, cudaStream_t stream)
{
    if (T <= 0)
	throw runtime_error("launch_s0_kernel: number of time samples T must be > 0");
    if (T & 127)
	throw runtime_error("launch_s0_kernel: number of time samples T must be a multiple of 128");
    if (F <= 0)
	throw runtime_error("launch_s0_kernel: number of frequency samples F must be > 0");
    if (S <= 0)
	throw runtime_error("launch_s0_kernel: number of stations S must be > 0");
    if (S & 127)
	throw runtime_error("launch_s0_kernel: number of stations S must be a multiple of 128");
    if (ds <= 0)
	throw runtime_error("launch_s0_kernel: downsampling factor 'ds' must be positive");
    if (ds & 1)
	throw runtime_error("launch_s0_kernel: downsampling factor 'ds' must be even");

    long Tds = T / ds;
    
    if (T != (Tds * ds))
	throw runtime_error("launch_s0_kernel: number of time samples T must be a multiple of downsampling factor 'ds'");

    dim3 nblocks, nthreads;
    gputils::assign_kernel_dims(nblocks, nthreads, S >> 2, (F+3) >> 2, Tds);

    s0_kernel <<< nblocks, nthreads, 0, stream >>>
	((uint4 *) s0, (const uint *) pl_mask, T, F, S, ds);
    
    CUDA_PEEK("s0_kernel launch");
}


// Helper for gputils::Array<> version of launch_s0_kernel().
template<typename T>
static void check_3d_array(const Array<T> &a, const char *name)
{
    if (a.ndim != 3)
	throw runtime_error("launch_s0_kernel: expected '" + string(name) + "' to be a 3-d array, got shape=" + a.shape_str());
    if (!a.is_fully_contiguous())
	throw runtime_error("launch_s0_kernel: array '" + string(name) + "' is not contiguous");
    if (!a.on_gpu())
	throw runtime_error("launch_s0_kernel: array '" + string(name) + "' is not on the GPU");
}


// Arguments:
//
//  - s0: uint32 array of shape (T/ds, F, S)
//  - pl_mask: uint64 array of shape (T/128, (F+3)//4, S/8).
//  - ds: Time downsampling factor. Must be multiple of 2.

void launch_s0_kernel(Array<uint> &s0, const Array<ulong> &pl_mask, long ds, cudaStream_t stream)
{
    check_3d_array(s0, "s0");
    check_3d_array(pl_mask, "pl_mask");

    long Tds = s0.shape[0];
    long F = s0.shape[1];
    long S = s0.shape[2];

    long T128 = pl_mask.shape[0];
    long Fds = pl_mask.shape[1];
    long Sds = pl_mask.shape[2];

    if ((Tds*ds != T128*128) || (Fds != ((F+3)/4)) || (S != (Sds*8)))
	throw runtime_error("launch_s0_kernel: s0.shape=" + s0.shape_str() + " and pl_mask.shape=" + pl_mask.shape_str() + " are inconsistent");

    launch_s0_kernel(s0.data, pl_mask.data, 128*T128, F, S, ds, stream);
}


}  // namespace n2k
