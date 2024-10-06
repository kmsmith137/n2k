#ifndef _N2K_PL_KERNELS_HPP
#define _N2K_PL_KERNELS_HPP

#include <gputils/Array.hpp>

namespace n2k {
#if 0
}  // editor auto-indent
#endif


extern void launch_correlate_pl_kernel(int *V_out, const ulong *pl_mask, long T, long F, long S, long Nds, cudaStream_t stream=0, bool debug=false);

// pl_mask shape = (T/64, F, S)
// V_out shape = (T/Nds, F, ntiles, 16, 16)
extern void launch_correlate_pl_kernel(gputils::Array<int> &V_out, const gputils::Array<ulong> &pl_mask, long Nds, cudaStream_t stream=0, bool debug=false);


} // namespace n2k

#endif // _N2K_PL_KERNELS_HPP
