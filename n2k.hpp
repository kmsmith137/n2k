#ifndef _N2K_HPP
#define _N2K_HPP

#include <gputils.hpp>


namespace n2k {
#if 0
}  // editor auto-indent
#endif


struct constants
{
    // Currently hardcoded
    static constexpr int num_stations = 1024;   // a "station" is a (dish,polarization) pair
    static constexpr int bit_depth = 4;         // correlator operates on signed (4+4) bit data
    static constexpr int nt_divisor = 256;      // nt_inner must be divisible by this (see below)

    // Visibility matrix strides (int32)
    static constexpr int vmat_kstride = 2;
    static constexpr int vmat_istride = 2 * 1024;
    static constexpr int vmat_fstride = 2 * 1024 * 1024;

    // E-array strides (int32)
    static constexpr int gmem_fstride = 256;  // FIXME rename
    
    // Block/thread counts
    static constexpr int warps_per_block = 8;
    static constexpr int threads_per_block = 32 * warps_per_block;
    static constexpr int threadblocks_per_freq = 36;

    // Shared memory layout (see overleaf; all strides are int32)
    static constexpr int shmem_t8_stride = 1;     // Delta(t)=8,16
    static constexpr int shmem_s1_stride = 4;     // Delta(s)=1,2,4
    static constexpr int shmem_s8_stride = 33;    // Delta(s)=8,16,32,64
    static constexpr int shmem_reim_stride = 16 * 33;
    static constexpr int shmem_ab_stride = 32 * 33;
    static constexpr int shmem_t32_stride = 64 * 33;   // Delta(t)=32,64
    static constexpr int shmem_nbytes = 8 * shmem_t32_stride * 4;   // 256 times, convert int32 -> bytes

    // These switches will artificially remove an important part of the processing, in order
    // to measure computational overhead of key steps (but making the kernel incorrect!)
    //
    //  input_shuffle - transpose input data from natural ordering to tensor core ordering
    //  output_shuffle - transpose visibility matrix to cache-friendly ordering before writing
    //  negate_4bit - minus sign which arises when multiplying complex numbers
    
    static constexpr bool artificially_remove_input_shuffle = false;
    static constexpr bool artificially_remove_output_shuffle = false;
    static constexpr bool artificially_remove_negate_4bit = false;
};


class Correlator
{
public:
    // The number of frequency channels (currently 16 per GPU in CHORD) must be
    // specified at construction.
    
    Correlator(int nfreq);

    // The launch() function below launches the correlator kernel.
    //
    //  - The number of stations (i.e. (dish,polarization) pairs) is hardcoded to 1024.
    //    This could be generalized, but right now I'm just concentrating on full CHORD.
    //
    //  - Similarly, the electric field input data is assumed to be signed (4+4) bit
    //    complex.
    //
    //  - The time cadence is controlled by two parameters, 'nt_outer' and 'nt_inner'.
    //    The total number of time samples is (nt_outer * nt_inner), and the visibility
    //    matrix is computed every 'nt_inner' samples.
    //
    //  - The 'e_in' array is
    //
    //       int8[nt_outer*nt_inner][nfreq][nstation]  with all axes contiguous    (*)
    // 
    //    where we represent complex (4+4) as int8, and nstation is hardcoded to 1024.
    //
    //    Note in (*) that the real/imaginary axis is fastest varying, followed by the
    //    station axis, and both these axes are contiguous. These assumptions would be
    //    a lot of work to change. On the other hand, it would be easy to swap the ordering
    //    of the time/frequency axes, or to allow non-contiguous strides for these axes.
    //
    //  - Currently the visibility matrix 'vis_out' is stored as
    //
    //      int32[nt_outer][nfreq][nstations][nstations][2]     (**)
    //
    //    where the last axis is Re/Im, all axes are contiguous, nstations=1024, and only visibilities
    //    (i,j) with (i < j) are computed. That is, we only compute the upper triangle of the visibility
    //    matrix. This is okay since the visibility matrix is Hermitian. However, note that the storage
    //    scheme (**) uses twice as much memory as necessary. This is something that I'll improve soon.
    //
    //  - 'nt_inner' must be a multiple of 256.
    //
    //  - We assume that 4-bit electric field samples are in the range [-7,7], i.e. the value (-8)
    //    is not allowed. Currently, if (-8) does arise, then the output of the computation will
    //    be incorrect (silently!). This behavior would be easy to change (possibly with a little
    //    extra computational cost).
    //
    //  - We define the visibility matrix as V_{ij} = sum_t E_{it} E_{jt}^*, with the complex
    //    conjugate on the second factor. (This would be trivial to change.)
    //
    //  - The kernel will segfault if run on a GPU which is not the cuda default device.
    //    This will be easy to fix later.
    //
    //  - The same Correlator object can be used to launch multiple kernels concurrently (either in
    //    serial using the same cuda stream, or in parallel using different cuda streams).
    //
    //  - The first few calls to launch() sometimes run slow, for reasons I haven't figured out yet.
    //    If you run the 'time-correlator' program, you can see the first few calls take longer, but
    //    the timing quickly settles down.

    void launch(int *vis_out, const int8_t *e_in, int nt_outer, int nt_inner,
		cudaStream_t stream=nullptr, bool sync=false) const;

    // This version of launch() uses gputils::Array objects instead of bare pointers.
    // Both arrays must be allocated on the GPU.
    //
    // The 'vis_out' array must have shape (nt_outer, nfreq, nstations, nstations, 2).
    // If nt_outer==1, then shape (nfreq, nstations, nstations, 2) is also okay.
    //
    // The 'e_in' array must have shape (nt_outer * nt_inner, nfreq, nstations).
    
    void launch(gputils::Array<int> &vis_out, const gputils::Array<int8_t> &e_in,
		int nt_outer, int nt_inner, cudaStream_t stream=nullptr, bool sync=false) const;
    
    // Initialized by constructor.
    const int nfreq;
    
    using kernel_t = void (*)(int *, const int8_t *, const int *, int);

protected:
    // This small (currently 27 KB) array will persist in GPU memory for the lifetime of the Correlator object.
    // Note that the shared_ptr destructor will call cudaFree().
    std::shared_ptr<int> precomputed_offsets;

    kernel_t kernel;
};


// Used internally by Correlator constructor
extern std::shared_ptr<int> precompute_offsets(int emat_tstride);
extern Correlator::kernel_t get_kernel(int emat_tstride);

// Used internally to "promote" compile-time argument to runtime argument.
void register_kernel(int emat_tstride, Correlator::kernel_t kernel);

    
}  // namespace n2k


#endif // _N2K_HPP
