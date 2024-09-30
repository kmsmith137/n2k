#include "../include/n2k/s012_kernels.hpp"
#include "../include/n2k/internals.hpp"

#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>
#include <iostream>

using namespace std;
using namespace gputils;
using namespace n2k;


static void test_s012_time_downsample(int Nds, int Tout, int F, int S)
{
    cout << "test_s012_time_downsample: Nds=" << Nds << ", Tout=" << Tout << ", F=" << F << ", S=" << S << endl;

    long Tin = Tout * Nds;
    Array<ulong> s_in = make_random_s012_array(Tin,F,S);  // shape (Tin,F,3,S)
    Array<ulong> s_out_cpu({Tout,F,3,S}, af_uhost | af_zero);
    Array<ulong> s_out_gpu({Tout,F,3,S}, af_gpu | af_guard);
        
    for (int tout = 0; tout < Tout; tout++)
	for (int tin = tout*Nds; tin < (tout+1)*Nds; tin++)
	    for (int f = 0; f < F; f++)
		for (int n = 0; n < 3; n++)
		    for (int s = 0; s < S; s++)
			s_out_cpu.at({tout,f,n,s}) += s_in.at({tin,f,n,s});

    Array<ulong> s_in_gpu = s_in.to_gpu();
    launch_s012_time_downsample_kernel(s_out_gpu, s_in_gpu, Nds);
    s_out_gpu = s_out_gpu.to_host();

    assert_arrays_equal(s_out_cpu, s_out_gpu, "cpu", "gpu", {"t","f","n","s"});
}


int main(int argc, char **argv)
{
    for (int n = 0; n < 100; n++) {
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(4, 100);
	test_s012_time_downsample(v[0], v[1], v[2], 32*v[3]);  // (Nds, Tout, F, S)
    }
    
    return 0;
}
