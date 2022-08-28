#include "n2k.hpp"
#include "n2k_kernel.hpp"

using namespace std;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


struct KernelTableEntry
{
    Correlator::kernel_t kernel = nullptr;
    bool shmem_attr_set = false;
};


static unordered_map<int,KernelTableEntry> kernel_table;
static mutex kernel_table_lock;


Correlator::kernel_t get_kernel(int emat_tstride)
{
    unique_lock<mutex> ul(kernel_table_lock);
    
    KernelTableEntry e = kernel_table[emat_tstride];
    if (!e.kernel) {
	stringstream ss;
	ss << "n2k: You have requested a value of 'nfreq' which is not supported."
	   << " Sadly, you will need to recompile with 'template_instantiations/kernel_" << emat_tstride << ".cu";
	throw runtime_error(ss.str());
    }
    
    // Reference for cudaFuncSetAttribute()
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g422642bfa0c035a590e4c43ff7c11f8d
    
    if (!e.shmem_attr_set) {
	CUDA_CALL(cudaFuncSetAttribute(
	    e.kernel, 
	    cudaFuncAttributeMaxDynamicSharedMemorySize,
	    constants::shmem_nbytes
	));
	e.shmem_attr_set = true;
    }

    return e.kernel;
}


void register_kernel(int emat_tstride, Correlator::kernel_t kernel)
{
    KernelTableEntry e;
    e.kernel = kernel;
    
    unique_lock<mutex> ul(kernel_table_lock);
    kernel_table[emat_tstride] = e;
}


}  // namespace n2k
