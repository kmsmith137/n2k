#include "n2k.hpp"
#include "n2k_kernel.hpp"

using namespace std;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


struct KernelTableEntry
{
    int nstations = 0;
    int emat_tstride = 0;
    mutable bool shmem_attr_set = false;
    Correlator::kernel_t kernel = nullptr;
};


static vector<KernelTableEntry> kernel_table;
static mutex kernel_table_lock;


Correlator::kernel_t get_kernel(int nstations, int emat_tstride)
{
    unique_lock<mutex> ul(kernel_table_lock);

    for (const auto &e: kernel_table) {
	if ((e.nstations != nstations) || (e.emat_tstride != emat_tstride))
	    continue;
    
	// Reference for cudaFuncSetAttribute()
	// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g422642bfa0c035a590e4c43ff7c11f8d
    
	if (!e.shmem_attr_set) {
	    CUDA_CALL(cudaFuncSetAttribute(
	        e.kernel, 
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		CorrelatorParams::shmem_nbytes
	    ));
	    e.shmem_attr_set = true;
	}

	return e.kernel;
    }
    
    stringstream ss;
    ss << "n2k: You have requested a value of 'nfreq' which is not supported."
       << " Sadly, you will need to recompile with 'template_instantiations/kernel_" << emat_tstride << ".cu";
    throw runtime_error(ss.str());
}


void register_kernel(int nstations, int emat_tstride, Correlator::kernel_t kernel)
{
    KernelTableEntry e;
    e.nstations = nstations;
    e.emat_tstride = emat_tstride;
    e.kernel = kernel;
    
    unique_lock<mutex> ul(kernel_table_lock);
    kernel_table.push_back(e);
}


}  // namespace n2k
