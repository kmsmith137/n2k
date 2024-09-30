#include "../include/n2k/s012_kernels.hpp"

#include <gputils/string_utils.hpp>
#include <gputils/test_utils.hpp>
#include <iostream>
#include <numeric>  // std::lcm()

using namespace std;
using namespace gputils;
using namespace n2k;


// These are the constraints needed by the GPU kernel.
// (Not all of these constraints are needed by the reference kernel.)

static void _check_args(long T, long F, long S, long ds)
{
    assert(T > 0);
    assert(F > 0);
    assert(S > 0);
    assert(ds > 0);

    assert((T % 128) == 0);
    assert((S % 128) == 0);
    assert((T % ds) == 0);
    assert((ds % 2) == 0);
}


// Input array:   ulong pl_mask[T/128, (F+3)/4, S/8]
// Output array:  ulong s0[T/ds, F, S]

static Array<ulong> reference_s0_kernel(const Array<ulong> &pl_mask, long T, long F, long S, long ds)
{
    _check_args(T, F, S, ds);
    
    long Tds = T/ds;
    long Fds = (F+3)/4;
    long Sds = S/8;
    
    assert(pl_mask.shape_equals({T/128, Fds, Sds}));
    
    Array<ulong> s0_arr({Tds, F, S}, af_rhost);
    
    for (long tds = 0; tds < Tds; tds++) {
	long t2_lo = tds * (ds >> 1);
	long t2_hi = (tds+1) * (ds >> 1);
    
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


static void test_s0_kernel(const Array<ulong> &pl_mask, long T, long F, long S, long ds)
{
    stringstream ss;
    ss << "test_s0_kernel(T=" << T << ", F=" << F << ", S=" << S << ", ds=" << ds << ")";
    cout << ss.str() << ": start" << endl;
    
    Array<ulong> s0_ref = reference_s0_kernel(pl_mask, T, F, S, ds);
    Array<ulong> s0_gpu(s0_ref.ndim, s0_ref.shape, af_gpu | af_guard);
    launch_s0_kernel(s0_gpu, pl_mask.to_gpu(), ds);
    
    gputils::assert_arrays_equal(s0_ref, s0_gpu, "cpu", "gpu", {"tds","f","s"});
    cout << ss.str() << ": pass" << endl;
}


static void test_s0_kernel(long T, long F, long S, long ds)
{
    Array<ulong> pl_mask({T/128, (F+3)/4, S/8}, af_rhost);
    ulong *pl = pl_mask.data;
    
    for (long i = 0; i < pl_mask.size; i++) {
	pl[i] = default_rng();
	pl[i] ^= (ulong(default_rng()) << 22);
	pl[i] ^= (ulong(default_rng()) << 44);
    }

    test_s0_kernel(pl_mask, T, F, S, ds);
}


static void test_s0_kernel()
{
    long ds = 2 * rand_int(1, 200);
    long Tdiv = std::lcm(ds, 128);

    // v = (T/Tdiv, F, S/128).
    vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, (1000*1000)/Tdiv);
    test_s0_kernel(v[0]*Tdiv, v[1], v[2]*128, ds);  // (T, F, S, ds)
}


int main(int argc, char **argv)
{
    // test_s0_kernel(128, 1, 256, 2);
    // test_s0_kernel(13568, 4, 128, 212);
    
    for (int i = 0; i < 100; i++)
	test_s0_kernel();
    
    return 0;
}
