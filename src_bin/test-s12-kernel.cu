#include "../include/n2k/s012_kernels.hpp"
#include "../include/n2k/internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>
#include <iostream>

using namespace std;
using namespace gputils;
using namespace n2k;


static void test_s12_kernel(int Nds, int Tout, int F, int S, int fstride)
{
    cout << "test_s12_kernel: Nds=" << Nds << ", Tout=" << Tout << ", F=" << F << ", S=" << S << ", fstride=" << fstride << endl;
    assert(fstride >= 2*S);

    long Tin = Tout * Nds;
    Array<complex<int>> e_cpu = make_random_unpacked_e_array(Tin,F,S);  // shape (Tin,F,S)
    Array<uint8_t> e_gpu = pack_e_array(e_cpu);
    e_gpu = e_gpu.to_gpu();
    
    Array<ulong> s_cpu({Tout,F,2,S}, af_uhost | af_zero);
    Array<ulong> s_gpu({Tout,F,2,S}, {F*fstride,fstride,S,1}, af_gpu | af_guard);

    launch_s12_kernel(s_gpu, e_gpu, Nds);
    CUDA_CALL(cudaDeviceSynchronize());

    // e_cpu -> s_cpu
    for (int tout = 0; tout < Tout; tout++) {
	for (int tin = tout*Nds; tin < (tout+1)*Nds; tin++) {
	    for (int f = 0; f < F; f++) {
		for (int s = 0; s < S; s++) {
		    complex<int> e = e_cpu.at({tin,f,s});
		    uint e2 = e.real()*e.real() + e.imag()*e.imag();
		    s_cpu.at({tout,f,0,s}) += e2;
		    s_cpu.at({tout,f,1,s}) += e2*e2;
		}
	    }
	}
    }

    assert_arrays_equal(s_cpu, s_gpu, "cpu", "gpu", {"t","f","n","s"});
}


int main(int argc, char **argv)
{
    // FIXME make this global, and use in many unit tests / asserts.
    const int max_stations = 4096;
    
    for (int n = 0; n < 100; n++) {
	int S = 128 * rand_int(1, (max_stations/128)+1);
	int fstride = 4 * rand_int(S/2, S+1);
	vector<ssize_t> v = gputils::random_integers_with_bounded_product(3, 400000/S);
	test_s12_kernel(v[0], v[1], v[2], S, fstride);  // (Nds,Tout,F,S,fstride)
    }
    
    return 0;
}
