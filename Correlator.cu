#include <gputils.hpp>
#include "n2k.hpp"

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


Correlator::Correlator(int nfreq_) : nfreq(nfreq_)
{
    int emat_tstride = nfreq * constants::gmem_fstride;
    
    this->precomputed_offsets = precompute_offsets(emat_tstride);
    this->kernel = get_kernel(emat_tstride);
}


void Correlator::launch(int *vis_out, const int8_t *e_in, int nt_outer, int nt_inner, cudaStream_t stream, bool sync) const
{
    assert(nt_outer > 0);
    assert(nt_inner > 0);
    assert(nt_inner % constants::nt_divisor == 0);
    assert(vis_out != nullptr);
    assert(e_in != nullptr);

    dim3 nblocks;
    nblocks.x = constants::threadblocks_per_freq;
    nblocks.y = nfreq;
    nblocks.z = nt_outer;
    
    int nthreads = constants::threads_per_block;
    int shmem_nbytes = constants::shmem_nbytes;
    const int *poffsets = this->precomputed_offsets.get();
    
    kernel <<<nblocks, nthreads, shmem_nbytes, stream >>> (vis_out, e_in, poffsets, nt_inner);
    CUDA_PEEK("Correlator::launch");

    if (sync)
	CUDA_CALL(cudaStreamSynchronize(stream));
}


void Correlator::launch(Array<int> &vis_out, const Array<int8_t> &e_in, int nt_outer, int nt_inner, cudaStream_t stream, bool sync) const
{
    int nt_expected = nt_outer * nt_inner;
    int nstat = constants::num_stations;
    
    if (!e_in.shape_equals({nt_expected,nfreq,nstat})) {
	stringstream ss;
	ss << "Correlator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer << ", nt_inner=" << nt_inner << ")"
	   << ": expected emat shape=(" << nt_expected << "," << nfreq << "," << nstat << ")"
	   << ", got shape=" << e_in.shape_str();
	throw runtime_error(ss.str());
    }
			   
    bool vflag1 = vis_out.shape_equals({nt_outer, nfreq, nstat, nstat, 2});
    bool vflag2 = vis_out.shape_equals({nfreq, nstat, nstat, 2});
    bool vshape_ok = vflag1 || (vflag2 && (nt_outer == 1));

    if (!vshape_ok) {
	stringstream ss;
	ss << "Correlator::launch(nfreq=" << nfreq << ", nt_outer=" << nt_outer << ", nt_inner=" << nt_inner << ")"
	   << ": expected vmat shape=(" << nt_outer << "," << nfreq << "," << nstat << "," << nstat << ",2" << ")";

	if (nt_outer == 1)
	    ss << " or shape=(" << nfreq << "," << nstat << "," << nstat << ",2" << ")";

	ss << ", got shape=" << vis_out.shape_str();
	throw runtime_error(ss.str());
    }

    assert(vis_out.is_fully_contiguous());
    assert(vis_out.on_gpu());
    assert(e_in.is_fully_contiguous());
    assert(e_in.on_gpu());
    
    this->launch(vis_out.data, e_in.data, nt_outer, nt_inner, stream, sync);
}


}  // namespace n2k
