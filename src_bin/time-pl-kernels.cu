#include "../include/n2k/pl_kernels.hpp"
#include <gputils/CudaStreamPool.hpp>

using namespace std;
using namespace gputils;
using namespace n2k;


static void time_correlate_pl(const string &name, long T, long F, long S, long Nds, long ninner)
{
    const long nouter = 10;
    
    long Tout = T / Nds;
    long ntiles = ((S/8) * ((S/8)+1)) / 2;

    Array<ulong> pl({T/64, F, S}, af_gpu | af_zero);
    Array<int> v({Tout,F,ntiles,8,8}, af_gpu | af_zero);
    double gb = 1.0e-9 * ninner * (8*pl.size + 4*v.size);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (long i = 0; i < ninner; i++)
	    launch_correlate_pl_kernel(v, pl, Nds, stream);  // calls CUDA_PEEK()
    };

    CudaStreamPool sp(callback, nouter, 1, name);
    sp.monitor_throughput("Global memory BW (GB/s)", gb);
    sp.run();
}


int main(int argc, char **argv)
{
    // (T,F) artificially increased.
    time_correlate_pl("CHORD pathfinder", 192*1024, 64*1024, 16, 128*48, 5);
    time_correlate_pl("Full CHORD", 48*1024, 32*1024, 128, 128*48, 5);
    
    return 0;
}
