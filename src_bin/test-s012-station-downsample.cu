#include "../include/n2k/launch_s0_kernel.hpp"
#include "../include/n2k/internals.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>

using namespace std;
using namespace gputils;
using namespace n2k;


static void test_s012_station_downsample(int T, int F, int S)
{
    cout << "test_s012_station_downsample: T=" << T << ", F=" << F << ", S=" << S << endl;

    Array<uint> s_in = make_random_s012_array(T,F,S);  // shape (T,F,3,S)
    Array<uint8_t> bf_mask = make_random_bad_feed_mask(S);
    
    Array<uint> s_out_cpu({T,F,3}, af_uhost | af_zero);
    Array<uint> s_out_gpu({T,F,3}, af_gpu | af_guard);

    for (int t = 0; t < T; t++)
	for (int f = 0; f < F; f++)
	    for (int n = 0; n < 3; n++)
		for (int s = 0; s < S; s++)
		    s_out_cpu.at({t,f,n}) += (bf_mask.data[s] ? s_in.at({t,f,n,s}) : 0);

    Array<uint> s_in_gpu = s_in.to_gpu();
    Array<uint8_t> bf_mask_gpu = bf_mask.to_gpu();
    launch_s012_station_downsample_kernel(s_out_gpu, s_in_gpu, bf_mask_gpu);
    CUDA_CALL(cudaDeviceSynchronize());
    
    assert_arrays_equal(s_out_cpu, s_out_gpu, "cpu", "gpu", {"t","f","n"});
}


int main(int argc, char **argv)
{
    // FIXME make this global, and use in many unit tests / asserts.
    const int max_stations = 4096;
    
    for (int n = 0; n < 100; n++) {
	int S = 128 * rand_int(1, (max_stations/128)+1);
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(2, 400000/S);
	test_s012_station_downsample(v[0], v[1], S);  // (T,F,S)
    }
    
    return 0;
}
