#include "../include/n2k/pl_kernels.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>

using namespace std;
using namespace n2k;
using namespace gputils;


// FIXME could be improved.
inline uint bit_count(ulong x)
{
    int ret = 0;
    
    for (int i = 0; i < 64; i++)
	ret += ((x & (1UL << i)) ? 1 : 0);

    return ret;
}


static void test_correlate_pl_mask(long T, long F, long S, long Nds)
{
    cout << "test_correlate_pl_mask: T=" << T << ", F=" << F << ", S=" << S << ", Nds=" << Nds << endl;

    long Tout = T / Nds;
    long ntiles = ((S/16) * ((S/16)+1)) / 2;
    
    Array<ulong> pl_cpu({T/64, F, S}, af_rhost | af_zero);
    Array<int> v_cpu({Tout,F,ntiles,16,16}, af_uhost);
    Array<int> v_gpu({Tout,F,ntiles,16,16}, af_gpu | af_guard);

    for (long i = 0; i < pl_cpu.size; i++) {
	ulong x = ulong(gputils::default_rng());
	x ^= (ulong(gputils::default_rng()) << 22);
	x ^= (ulong(gputils::default_rng()) << 44);
	pl_cpu.data[i] = x;
    }

    Array<ulong> pl_gpu = pl_cpu.to_gpu();
    launch_correlate_pl_kernel(v_gpu, pl_gpu, Nds, 0, true);   // stream=0, debug=true
    CUDA_CALL(cudaDeviceSynchronize());

    // CPU implementation of correlate_pl_kernel() starts here.

    int N64 = Nds / 64;
    int t64_stride = F*S;
    
    for (long tout = 0; tout < Tout; tout++) {
	for (long f = 0; f < F; f++) {
	    const ulong *pl_tf = &pl_cpu.at({tout*N64,f,0});  // shape (N64,S), strides (t64_stride, 1)
	    int *v_tf = &v_cpu.at({tout,f,0,0,0});            // shape (ntiles,16,16), contiguous
	    
	    for (long ixtile = 0; ixtile < (S/16); ixtile++) {
		for (long iytile = 0; iytile <= ixtile; iytile++) {
		    long itile = (ixtile*(ixtile+1))/2 + iytile;
		    const ulong *plx = pl_tf + 16*ixtile;   // shape (N64,16), strides (t64_stride, 1)
		    const ulong *ply = pl_tf + 16*iytile;   // shape (N64,16), strides (t64_stride, 1)
		    int *vtile = v_tf + 256*itile;          // shape (16,16), contiguous

		    for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
			    int v = 0;
			    for (int t64 = 0; t64 < N64; t64++) {
				long x = plx[t64*t64_stride + i];
				long y = ply[t64*t64_stride + j];
				v += bit_count(x & y);
			    }
			    vtile[16*i+j] = v;
			}
		    }
		}
	    }
	}
    }

    gputils::assert_arrays_equal(v_cpu, v_gpu, "cpu", "gpu", {"tout","f","tile","i","j"});
}


static void test_correlate_pl_mask()
{
    for (int n = 0; n < 100; n++) {
	// v = (T/Nds, F, Nds/128)
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, 1000);

	long Nds = v[2] * 128;
	long F = v[1];
	long T = v[0] * Nds;
	long S = 16;   // for now
	
	test_correlate_pl_mask(T, F, S, Nds);
    }
}


int main(int argc, char **argv)
{
    test_correlate_pl_mask();
    return 0;
}
