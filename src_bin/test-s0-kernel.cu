#include "../include/n2k/s012_kernels.hpp"

#include <gputils/string_utils.hpp>
#include <gputils/test_utils.hpp>
#include <iostream>
#include <numeric>  // std::lcm()

using namespace std;
using namespace gputils;
using namespace n2k;


// Input array:   ulong pl_mask[T/128, (F+3)/4, S/8]
// Output array:  ulong s0[T/Nds, F, S]

static Array<ulong> reference_s0_kernel(const Array<ulong> &pl_mask, long T, long F, long S, long Nds)
{
    long Tds = T/Nds;
    long Fds = (F+3)/4;
    long Sds = S/8;
    
    assert(pl_mask.shape_equals({T/128, Fds, Sds}));
    
    Array<ulong> s0_arr({Tds, F, S}, af_rhost);
    
    for (long tds = 0; tds < Tds; tds++) {
	long t2_lo = tds * (Nds >> 1);
	long t2_hi = (tds+1) * (Nds >> 1);
    
	for (long fds = 0; fds < Fds; fds++) {
	    for (long sds = 0; sds < Sds; sds++) {
		ulong s0_elt = 0;

		for (long t2 = t2_lo; t2 < t2_hi; t2++) {
		    ulong x = pl_mask.at({t2 >> 6, fds, sds});
		    ulong bit = 1UL << (t2 & 63);
		    s0_elt += (x & bit) ? 2 : 0;
		}

		for (long f = 4*fds; f < min(4*fds+4,F); f++)
		    for (long s = 8*sds; s < 8*sds+8; s++)
			s0_arr.at({tds, f, s}) = s0_elt;
	    }
	}
    }

    return s0_arr;
}


static void test_s0_kernel(const Array<ulong> &pl_mask, long T, long F, long S, long Nds, long fstride)
{
    stringstream ss;
    ss << "test_s0_kernel(T=" << T << ", F=" << F << ", S=" << S << ", Nds=" << Nds << ", fstride=" << fstride << ")";
    cout << ss.str() << endl;
    
    Array<ulong> s0_ref = reference_s0_kernel(pl_mask, T, F, S, Nds);
    Array<ulong> s0_gpu({T/Nds,F,S}, {F*fstride,fstride,1}, af_gpu | af_guard);
    Array<ulong> pl_mask_gpu = pl_mask.to_gpu();
    launch_s0_kernel(s0_gpu, pl_mask_gpu, Nds);
    
    gputils::assert_arrays_equal(s0_ref, s0_gpu, "cpu", "gpu", {"tds","f","s"});
}


static void test_s0_kernel(long T, long F, long S, long Nds, long fstride)
{
    Array<ulong> pl_mask({T/128, (F+3)/4, S/8}, af_rhost);
    ulong *pl = pl_mask.data;
    
    for (long i = 0; i < pl_mask.size; i++) {
	pl[i] = default_rng();
	pl[i] ^= (ulong(default_rng()) << 22);
	pl[i] ^= (ulong(default_rng()) << 44);
    }

    test_s0_kernel(pl_mask, T, F, S, Nds, fstride);
}


static void test_s0_kernel()
{
    long Nds = 2 * rand_int(1, 200);
    long Tdiv = std::lcm(Nds, 128);

    vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, (1000*1000)/Tdiv);
    long T = v[0]*Tdiv;
    long F = v[1];
    long S = v[2]*128;
    
    long fstride = 4 * rand_int(S/4, S+1);
    test_s0_kernel(T, F, S, Nds, fstride);
}


int main(int argc, char **argv)
{
    // test_s0_kernel(128, 1, 256, 2);
    // test_s0_kernel(13568, 4, 128, 212);
    
    for (int i = 0; i < 100; i++)
	test_s0_kernel();
    
    return 0;
}
