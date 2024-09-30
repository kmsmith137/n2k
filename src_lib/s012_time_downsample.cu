#include "../include/n2k/s012_kernels.hpp"
#include "../include/n2k/internals.hpp"

#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// Kernel arguments:
//
//   uint   S012_out[T/Nds][M];   // where M is number of "spectator" indices (3*F*S)
//   uint   S012_in[T][M]; 
//   long   Nds;
//   long   Tout;
//   long   M;
//
// Parallelization:
//
//   - Each thread does one (output time, spectator index).
//
//   - Index mapping is:
//
//      {threadIdx,blockIdx}.x <-> spectator index
//      {threadIdx,blockIdx}.y <-> output (downsampled) time

__global__ void s012_time_downsample_kernel(uint *Sout, const uint *Sin, int Nds, int Tout, int M)
{
    // Per-thread (downsampled time, spectator index)
    uint m = (blockIdx.x * blockDim.x) + threadIdx.x;
    ulong tout = (blockIdx.y * blockDim.y) + threadIdx.y;
    bool valid = (m < M) && (tout < Tout);

    // Per-thread base indices and pointer shift.
    ulong in_base = (M*Nds)*tout + m;
    ulong out_base = M*tout + m;
    Sin += (valid ? in_base : 0);
    
    uint s = 0;
    for (int n = 0; n < Nds; n++)
	s += Sin[n*M];
    
    if (valid)
	Sout[out_base] = s;
}

// launch_s012_time_downsample_kernel(): bare-pointer interface.
//
//   uint   S012_out[T/Nds][M];   // where M is number of "spectator" indices (3*F*S)
//   uint   S012_in[T][M];
//   long   Nds;
//   long   Tout;
//   long   M;

void launch_s012_time_downsample_kernel(uint *Sout, const uint *Sin, long Nds, long Tout, long M, cudaStream_t stream)
{
    int threads_per_block = 128;
    bool noisy = false;

    if ((Sout == nullptr) || (Sin == nullptr))
	throw runtime_error("launch_s012_time_downsample_kernel(): data pointer was NULL");
    if (Nds <= 0)
	throw runtime_error("launch_s012_time_downsample_kernel(): expected Nds > 0");
    if (Tout <= 0)
	throw runtime_error("launch_s012_time_downsample_kernel(): expected Tout > 0");
    if (M <= 0)
	throw runtime_error("launch_s012_time_downsample_kernel(): expected M > 0");
    
    // Necessary for efficiency (but not correctness) of kernel.
    // Always satisfied for CHORD, CHIME, HIRAX.
    if (M % 32)
	throw runtime_error("launch_s012_time_downsample_kernel(): expected M to be a multple of 32");
    
    dim3 nblocks, nthreads;
    gputils::assign_kernel_dims(nblocks, nthreads, M, Tout, 1, threads_per_block, noisy);

    s012_time_downsample_kernel <<< nblocks, nthreads, 0, stream >>>
	(Sout, Sin, Nds, Tout, M);

    CUDA_PEEK("launch s012_time_downsample");
}


// launch_s012_time_downsample_kernel(): array interface
//
//   uint   S012_out[T/Nds][F][3][S];   // S must be multiple of 32
//   uint   S012_in[T][F][3][S];
//   long   Nds;

void launch_s012_time_downsample_kernel(Array<uint> &Sout, const Array<uint> &Sin, long Nds, cudaStream_t stream)
{
    check_array(Sout, "launch_s012_time_downsample_kernel", "Sout", 4, true);  // ndim=4, contiguous=true
    check_array(Sin, "launch_s012_time_downsample_kernel", "Sin", 4, true);    // ndim=4, contiguous=true

    if (Sin.shape[0] != Sout.shape[0] * Nds)
	throw runtime_error("launch_s012_time_downsample_kernel(): inconsistent number of time samples in input/output arrays");
    if (Sin.shape[1] != Sout.shape[1])
	throw runtime_error("launch_s012_time_downsample_kernel(): inconsistent number of frequency channels in input/output arrays");
    if (Sout.shape[2] != 3)
	throw runtime_error("launch_s012_time_downsample_kernel(): expected Sout.shape[2] == 3");
    if (Sin.shape[2] != 3)
	throw runtime_error("launch_s012_time_downsample_kernel(): expected Sin.shape[2] == 3");
    if (Sin.shape[3] != Sout.shape[3])
	throw runtime_error("launch_s012_time_downsample_kernel(): inconsistent number of stations in input/output arrays");

    long Tout = Sout.shape[0];
    long M = 3 * Sout.shape[1] * Sout.shape[3];
    
    launch_s012_time_downsample_kernel(Sout.data, Sin.data, Nds, Tout, M, stream);
}


}  // namespace n2k
