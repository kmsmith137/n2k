#include "../include/n2k/internals.hpp"
#include "../include/n2k/interpolation.hpp"
#include "../include/n2k/bad_feed_mask.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>

using namespace std;
using namespace n2k;
using namespace gputils;


// -------------------------------------------------------------------------------------------------
//
// test_pack_e_array()


static void test_pack_e_array(int T, int F, int S)
{
    cout << "test_pack_e_array: T=" << T << ", F=" << F << ", S=" << S << endl;
    
    Array<complex<int>> E_src = make_random_unpacked_e_array(T,F,S);
    Array<uint8_t> E_packed = pack_e_array(E_src);
    Array<complex<int>> E_dst = unpack_e_array(E_packed);

    assert_arrays_equal(E_src, E_dst, "E_src", "E_dst", {"t","f","s"});
}


// -------------------------------------------------------------------------------------------------
//
// test_transpose_bit_with_lane()


static void reference_transpose_bit_with_lane(uint dst[32], const uint src[32], uint bit, uint lane)
{
    assert((bit == 1) || (bit == 2) || (bit == 4) || (bit == 8) || (bit == 16));
    assert((lane == 1) || (lane == 2) || (lane == 4) || (lane == 8) || (lane == 16));

    for (uint ldst = 0; ldst < 32; ldst++) {
	dst[ldst] = 0;
	
	for (uint bdst = 0; bdst < 32; bdst++) {
	    uint lsrc = ldst & ~lane;
	    uint bsrc = bdst & ~bit;

	    if (ldst & lane)
		bsrc |= bit;
	    if (bdst & bit)
		lsrc |= lane;
	    
	    if (src[lsrc] & (1U << bsrc))
		dst[ldst] |= (1U << bdst);
	}
    }
}


// Launch with nblocks = nwarps = 1.
__global__ void transpose_bit_with_lane_kernel(uint *dst, const uint *src, uint bit, uint lane)
{
    uint x = src[threadIdx.x];

    if (bit == 1)
	x = transpose_bit_with_lane<1> (x, lane);
    else if (bit == 2)
	x = transpose_bit_with_lane<2> (x, lane);
    else if (bit == 4)
	x = transpose_bit_with_lane<4> (x, lane);
    else if (bit == 8)
	x = transpose_bit_with_lane<8> (x, lane);
    else if (bit == 16)
	x = transpose_bit_with_lane<16> (x, lane);
	
    dst[threadIdx.x] = x;
}


static void test_transpose_bit_with_lane(uint bit, uint lane)
{
    cout << "test_transpose_bit_with_lane: bit=" << bit << ", lane=" << lane << endl;
    std::mt19937 &rng = gputils::default_rng;

    Array<uint> src({32}, af_rhost);
    for (int i = 0; i < 32; i++)
	src.data[i] = uint(rng()) ^ (uint(rng()) << 16);
    
    Array<uint> dst_cpu({32}, af_uhost);
    reference_transpose_bit_with_lane(dst_cpu.data, src.data, bit, lane);

    Array<uint> src_gpu = src.to_gpu();
    Array<uint> dst_gpu({32}, af_gpu);
    transpose_bit_with_lane_kernel <<<1,32>>> (dst_gpu.data, src_gpu.data, bit, lane);
    CUDA_PEEK("transpose_bit_with_lane_kernel launch");

    assert_arrays_equal(dst_cpu, dst_gpu, "cpu", "gpu", {"i"});
}


static void test_transpose_bit_with_lane()
{
    for (uint bit = 1; bit <= 16; bit *= 2)
	for (uint lane = 1; lane <= 16; lane *= 2)
	    test_transpose_bit_with_lane(bit, lane);
}


// -------------------------------------------------------------------------------------------------
//
// test_load_bad_feed_mask()


__global__ void bad_feed_mask_kernel(uint *out, const uint8_t *bf_mask, int S)
{
    // Length max(S/32,32).
    extern __shared__ uint shmem[];

    int t0 = threadIdx.z;
    t0 = (t0 * blockDim.y) + threadIdx.y;
    t0 = (t0 * blockDim.x) + threadIdx.x;

#if 0
    int nt = blockDim.z * blockDim.y * blockDim.x;
    for (int t = t0; t < nt; t += nt)
	shmem[t] = 0;
    __syncthreads();
#endif

    out[t0] = load_bad_feed_mask((const uint *) bf_mask, shmem, S);
}


static void test_bad_feed_mask(int S, uint Wx, uint Wy, uint Wz, int s0=-1)
{
    cout << "test_bad_feed_mask: S=" << S << ", Wx=" << Wx << ", Wy=" << Wy << ", Wz=" <<  Wz;
    if (s0 >= 0)
	cout << ", s0=" << s0;
    cout << endl;
    
    assert(S > 0);
    assert(S <= 1024*Wx);
    assert((S % 128) == 0);
    
    Array<uint8_t> bf_mask({S}, af_rhost | af_zero);

    if (s0 >= 0) {
	assert(s0 < S);
	bf_mask.data[s0] = 1;
    }
    else {
	for (int s = 0; s < S; s++)
	    bf_mask.data[s] = rand_int(0,2);
    }

    int nt = 32 * Wx * Wy * Wz;
    int shmem_nbytes = 4 * max(S/32,32);
    Array<uint> out({nt}, af_gpu);
    
    Array<uint8_t> bf_gpu = bf_mask.to_gpu();
    bad_feed_mask_kernel <<< 1, {32*Wx,Wy,Wz}, shmem_nbytes, 0 >>> (out.data, bf_gpu.data, S);
    CUDA_PEEK("bad_feed_mask_kernel launch");
    out = out.to_host();

    for (uint iyz = 0; iyz < Wy*Wz; iyz++) {
	for (uint ix = 0; ix < 32*Wx; ix++) {
	    int bit = 0;
	    for (int s = ix; s < S; s += 32*Wx) {
		bool bf_cpu = bf_mask.data[s];
		bool bf_gpu = out.data[ix] & (1U << bit);
		
		if (bf_cpu != bf_gpu) {
		    cout << "failed! iyz=" << iyz << ", ix=" << ix << ", s=" << s << ", bit=" << "bf_cpu=" << bf_cpu << ", bf_gpu=" << bf_gpu << endl;
		    exit(1);
		}
		    
		bit++;
	    }
	}
    }
}


static void test_bad_feed_mask()
{
    for (int i = 0; i < 100; i++) {
	vector<ssize_t> W = random_integers_with_bounded_product(3, 32);
	int m = 8 * min(W[0],4L);
	int S = 128 * rand_int(1,m+1);
	test_bad_feed_mask(S, W[0], W[1], W[2]);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Test load_sigma_coeffs() in interpolation.hpp


// Launch with 1 block and T threads.
//  out: shape (T,4)
//  sigma_coeffs: shape (N,8)
//  ix: shape (T,)

__global__ void sigma_coeffs_test_kernel(float *out, const float *sigma_coeffs, const int *ix)
{
    int t = threadIdx.x;
    load_sigma_coeffs<true> (sigma_coeffs, ix[t], out[4*t], out[4*t+1], out[4*t+2], out[4*t+3]);   // Debug=true
}


static void test_load_sigma_coeffs(int T, int N)
{
    cout << "test_load_sigma_coeffs(T=" << T << ", N=" << N << ")" << endl;
    
    Array<float> out({T,4}, af_uhost);
    Array<float> coeffs({N,8}, af_rhost);
    Array<int> ix({T}, af_rhost);

    for (int i = 0; i < N; i++) {
	float x = rand_uniform();
	for (int j = 0; j < 8; j++)
	    coeffs.at({i,j}) = x;
    }
    
    for (int t = 0; t < T; t++) {
	int i = rand_int(0, N-3);
	ix.at({t}) = i;
	
	for (int j = 0; j < 4; j++)
	    out.at({t,j}) = coeffs.at({i+j,0});
    }

    Array<float> out_gpu({T,4}, af_gpu | af_zero);
    Array<float> coeffs_gpu = coeffs.to_gpu();
    Array<int> ix_gpu = ix.to_gpu();

    sigma_coeffs_test_kernel<<<1,T>>> (out_gpu.data, coeffs_gpu.data, ix_gpu.data);
    CUDA_PEEK("sigma_coeffs_test_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(out, out_gpu, "cpu", "gpu", {"t","j"});
}


static void test_load_sigma_coeffs()
{
    for (int i = 0; i < 10; i++) {
	int T = 32 * rand_int(1,10);
	int N = rand_int(10, 100);
	test_load_sigma_coeffs(T, N);
    }
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    test_pack_e_array(16, 32, 128);
    test_transpose_bit_with_lane();
    test_bad_feed_mask();
    test_load_sigma_coeffs();
    return 0;
}
