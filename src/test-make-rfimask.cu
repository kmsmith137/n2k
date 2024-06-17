#include "../include/n2k.hpp"
#include <numeric>  // std::lcm()

using namespace std;
using namespace gputils;
using namespace n2k;


// float sk_sigma[nbits/ds];    // input array
// uint rfimask[nbits/32];      // output array


static Array<uint> reference_make_rfimask(const Array<float> &sk_sigma, long ds, float thresh)
{
    static constexpr uint ALL_ONES = 0xffffffffU;

    assert(sk_sigma.ndim == 1);
    assert(sk_sigma.is_fully_contiguous());
    assert(sk_sigma.on_host());
    
    assert(ds > 0);
    assert((ds % 32) == 0);
    
    long nbits = ds * sk_sigma.size;  // nbits is divisible by 32, since ds is divisible by 32
    Array<uint> rfimask({nbits/32}, af_rhost);
    
    const float *src = sk_sigma.data;
    uint *dst = rfimask.data;
    
    for (long idst = 0; idst < nbits/32; idst++) {
	long isrc = (32*idst) / ds;
	dst[idst] = (src[isrc] < thresh) ? ALL_ONES : 0;
    }

    return rfimask;
}


static void test_rfimask(const Array<float> &sk_sigma, long ds, float thresh)
{
    stringstream ss;
    ss << "test_rfimask(nbits=" << (ds*sk_sigma.size) << ", ds=" << ds << ", thresh=" << thresh << ")";
    cout << ss.str() << ": start" << endl;
	
    Array<uint> rfimask_cpu = reference_make_rfimask(sk_sigma, ds, thresh);
    
    Array<uint> rfimask_gpu({rfimask_cpu.size}, af_gpu | af_guard);
    launch_rfimask_maker(rfimask_gpu, sk_sigma.to_gpu(), ds, thresh);

    gputils::assert_arrays_equal(rfimask_cpu, rfimask_gpu, "cpu", "gpu", {"i"});
    cout << ss.str() << ": pass" << endl;
}



static void test_rfimask(long nbits, long ds, float thresh)
{
    assert(ds > 0);
    assert(nbits > 0);
    assert((ds % 32) == 0);
    assert((nbits % ds) == 0);
    assert((nbits % 128) == 0);

    Array<float> sk_sigma({nbits/ds}, af_rhost);
    float *sk = sk_sigma.data;

    for (long i = 0; i < sk_sigma.size; i++) {
	bool below = rand_int(0,2);
	double lo = below ? (thresh-4.0) : (thresh-0.01);
	double hi = below ? (thresh+0.1) : (thresh+4.0);
	sk[i] = rand_uniform(lo, hi);
    }

    test_rfimask(sk_sigma, ds, thresh);
}


static void test_rfimask()
{
    long ds = 32 * rand_int(1,10);
    long m = std::lcm(ds, 128);
    long nbits = m * rand_int(1, (1000*1000)/m);
    float thresh = rand_uniform(2.0, 5.0);
    
    test_rfimask(nbits, ds, thresh);
}


int main(int argc, char **argv)
{
    for (int i = 0; i < 100; i++)
	test_rfimask();
    return 0;
}
