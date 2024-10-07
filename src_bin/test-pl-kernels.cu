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


inline uint rand_uint()
{
    uint x = uint(gputils::default_rng());
    x ^= (uint(gputils::default_rng()) << 16);
    return x;
}


inline ulong rand_ulong()
{
    ulong x = ulong(gputils::default_rng());
    x ^= (ulong(gputils::default_rng()) << 22);
    x ^= (ulong(gputils::default_rng()) << 44);
    return x;
}


static void test_correlate_pl_mask(long T, long F, long S, long Nds, long rfimask_fstride)
{
    cout << "test_correlate_pl_mask: T=" << T << ", F=" << F << ", S=" << S
	 << ", Nds=" << Nds << ", rfimask_fstride=" << rfimask_fstride << endl;
    
    assert(rfimask_fstride >= T/32);

    long Tout = T / Nds;
    long ntiles = ((S/8) * ((S/8)+1)) / 2;
    
    Array<ulong> pl_cpu({T/64,F,S}, af_rhost | af_zero);
    Array<uint> rfimask_cpu({F,T/32}, {rfimask_fstride,1}, af_rhost | af_zero);
    Array<int> v_cpu({Tout,F,ntiles,8,8}, af_uhost);
    Array<int> v_gpu({Tout,F,ntiles,8,8}, af_gpu | af_guard);

    for (long i = 0; i < pl_cpu.size; i++)
	pl_cpu.data[i] = rand_ulong();

    for (long f = 0; f < F; f++)
	for (long t32 = 0; t32 < T/32; t32++)
	    rfimask_cpu.data[f*rfimask_fstride + t32] = rand_uint();
    
    Array<ulong> pl_gpu = pl_cpu.to_gpu();
    Array<uint> rfimask_gpu = rfimask_cpu.to_gpu();
    launch_correlate_pl_kernel(v_gpu, pl_gpu, rfimask_gpu, Nds);
    CUDA_CALL(cudaDeviceSynchronize());

    // CPU implementation of correlate_pl_kernel() starts here.

    int N64 = Nds / 64;
    int t64_stride = F*S;
    
    for (long tout = 0; tout < Tout; tout++) {
	for (long f = 0; f < F; f++) {
	    const ulong *pl_tf = &pl_cpu.at({tout*N64,f,0});  // shape (N64,S), strides (t64_stride, 1)
	    const ulong *rfi_tf = (const ulong *) &rfimask_cpu.at({f,tout*N64*2});      // shape (N64,)
	    int *v_tf = &v_cpu.at({tout,f,0,0,0});                    // shape (ntiles,8,8), contiguous
	    
	    for (long ixtile = 0; ixtile < (S/8); ixtile++) {
		for (long iytile = 0; iytile <= ixtile; iytile++) {
		    long itile = (ixtile*(ixtile+1))/2 + iytile;
		    const ulong *plx = pl_tf + 8*ixtile;   // shape (N64,8), strides (t64_stride, 1)
		    const ulong *ply = pl_tf + 8*iytile;   // shape (N64,8), strides (t64_stride, 1)
		    int *vtile = v_tf + 64*itile;          // shape (8,8), contiguous

		    for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
			    int v = 0;
			    for (int t64 = 0; t64 < N64; t64++) {
				ulong x = plx[t64*t64_stride + i];
				ulong y = ply[t64*t64_stride + j];
				ulong rfi = rfi_tf[t64];
				v += bit_count(x & y & rfi);
			    }
			    vtile[8*i+j] = v;
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
	long S = rand_int(0,2) ? 16 : 128;  // for now
	
	// v = (T/Nds, F, Nds/128)
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, 10*1000*1000/(S*S));
	long Nds = v[2] * 128;
	long F = v[1];
	long T = v[0] * Nds;
	long rfimask_fstride = rand_int(T/32, T/16);
	
	test_correlate_pl_mask(T, F, S, Nds, rfimask_fstride);
    }
}


int main(int argc, char **argv)
{
    test_correlate_pl_mask();
    return 0;
}
