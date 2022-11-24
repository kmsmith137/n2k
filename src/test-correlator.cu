#include <complex>
#include <iostream>
#include <gputils.hpp>

#include "../include/n2k.hpp"
#include "../include/n2k_kernel.hpp"

using namespace std;
using namespace gputils;
using namespace n2k;


// -------------------------------------------------------------------------------------------------
//
// unpack_4bit(), pack_4bit()


// Slow, intended only for testing!
// Given an array of shape S, unpack int32 -> int4[8] and return an array of shape S+(8,)
__host__ Array<int> unpack_4bit(const Array<int> &src)
{
    assert(src.ndim >= 1);
    
    vector<ssize_t> dst_shape(src.ndim+1, 8);
    for (int d = 0; d < src.ndim; d++)
	dst_shape[d] = src.shape[d];

    Array<int> dst(dst_shape);

    for (auto ix = dst.ix_start(); dst.ix_valid(ix); dst.ix_next(ix)) {
	// Extract 4 bits from x, starting at bit b.
	int x = src.at(src.ndim, &ix[0]);
	int b = ix[src.ndim] * 4;
	dst.at(ix) = (x << (28-b)) >> 28;
    }

    return dst;
}


// Slow, intended only for testing!
// Given an array of shape S+(8,), pack int4[8] -> int32 and return an array of shape S.
__host__ Array<int> pack_4bit(const Array<int> &src)
{
    assert(src.ndim >= 2);
    assert(src.shape[src.ndim-1] == 8);

    Array<int> dst(src.ndim-1, &src.shape[0], af_zero);
    
    for (auto ix = src.ix_start(); src.ix_valid(ix); src.ix_next(ix)) {
	// Extract 4 bits from x, starting at bit b.
	int x = src.at(ix);
	int b = ix[src.ndim-1] * 4;
	
	assert ((x >= -8) && (x <= 7));
	dst.at(dst.ndim, &ix[0]) |= ((x & 0xf) << b);
    }

    return dst;
}


__host__ void test_pack_unpack()
{
    int n = 100;
    Array<int> a({n}, af_random);
    Array<int> b = unpack_4bit(a);
    Array<int> c = pack_4bit(b);
    
    assert(b.shape_equals({n,8}));
    assert(c.shape_equals({n}));

#if 0
    for (int i = 0; i < n; i++) {
	cout << "  test_pack_unpack:";
	for (int j = 0; j < 8; j++)
	    cout << "  " << b.at({i,j});
	cout << "\n";
    }
#endif

    for (int i = 0; i < n; i++)
	assert(a.at({i}) == c.at({i}));

    cout << "test_pack_unpack: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// negate_4bit() testing


__global__ void negate_4bit_kernel(int *buf, int nsites, int ninner)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    
    for (int i = threadId; i < nsites; i += nthreads) {
	int x = buf[i];
	int y = 0;

	for (int j = 0; j < ninner; j++) {
	    x = CorrelatorKernel<128,32>::negate_4bit(x);
	    y |= x;   // This kludge is needed to prevent the compiler from optimizing away the loop.
	}

	buf[i] = y;
    }
}


__host__ void test_negate_4bit()
{
    const int nsites = 512;
    
    // negate_4bit() assumes its 4-bit input values are in the range [-7,8),
    // i.e. it fails for -8. The loop below generates random input data
    // satisfying this constraint.

    Array<int> src({nsites,8});    
    for (int i = 0; i < nsites; i++)
	for (int j = 0; j < 8; j++)
	    src.at({i,j}) = rand_int(-7, 8);

    Array<int> dst = pack_4bit(src).to_gpu();
    
    negate_4bit_kernel <<<nsites/32, 32>>> (dst.data, nsites, 1);
    CUDA_PEEK("negate_4bit_kernel (test)");
    CUDA_CALL(cudaDeviceSynchronize());

    dst = unpack_4bit(dst.to_host());

    for (int i = 0; i < nsites; i++)
	for (int j = 0; j < 8; j++)
	    assert(dst.at({i,j}) == -src.at({i,j}));

    cout << "test_negate_4bit: pass" << endl;
}



// -------------------------------------------------------------------------------------------------
//
// transpose_rank8_4bit() testing


__global__ void transpose_rank8_4bit_kernel(int *buf, int nsites, int ninner)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;

    for (int i = threadId; i < nsites; i += nthreads) {
	int x[8];

	#pragma unroll
	for (int j = 0; j < 8; j++)
	    x[j] = buf[8*i + j];

	for (int j = 0; j < ninner; j++)
	    CorrelatorKernel<128,32>::transpose_rank8_4bit(x);

	#pragma unroll
	for (int j = 0; j < 8; j++)
	    buf[8*i+j] = x[j];
    }
}


__host__ void test_transpose_rank8_4bit()
{
    const int nsites = 512;

    Array<int> a({nsites,8}, af_random);
    Array<int> src = unpack_4bit(a);
    
    a = a.to_gpu();
    transpose_rank8_4bit_kernel <<<nsites/32, 32>>> (a.data, nsites, 1);
    CUDA_PEEK("transpose_rank8_4bit_kernel");
    CUDA_CALL(cudaDeviceSynchronize());
    
    a = a.to_host();
    Array<int> dst = unpack_4bit(a);

    for (int i = 0; i < nsites; i++)
	for (int j = 0; j < 8; j++)
	    for (int k = 0; k < 8; k++)
		assert(src.at({i,j,k}) == dst.at({i,k,j}));

    cout << "test_transpose_rank8_4bit: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


__host__ int8_t pack_complex44(complex<int> z)
{
    assert((z.real() >= -7) && (z.real() <= 7));
    assert((z.imag() >= -7) && (z.imag() <= 7));

    if constexpr (CorrelatorParams::real_part_in_low_bits)
	return (z.real() & 0xf) | (z.imag() << 4);
    else
	return (z.imag() & 0xf) | (z.real() << 4);
}


__host__ void minimal_correlator_test(int nstations, int nfreq, int f, int sa, int sb, int t, complex<int> za, complex<int> zb)
{
    // Currently hardcoded
    const int nt_outer = 4;
    const int nt_inner = 1024;
    const int nt_tot = nt_outer * nt_inner;
    const int touter = int(t / nt_inner);
    
    assert(!CorrelatorParams::artificially_remove_input_shuffle);
    assert(!CorrelatorParams::artificially_remove_output_shuffle);
    assert(!CorrelatorParams::artificially_remove_negate_4bit);
    
    assert((f >= 0) && (f < nfreq));
    assert((sa >= 0) && (sa < nstations));
    assert((sb >= 0) && (sb < nstations));
    assert((t >= 0) && (t < nt_outer*nt_inner));
    assert((za.real() >= -7) && (za.real() <= 7));
    assert((za.imag() >= -7) && (za.imag() <= 7));
    assert((zb.real() >= -7) && (zb.real() <= 7));
    assert((zb.imag() >= -7) && (zb.imag() <= 7));
    
    assert(sa <= sb);
    assert((sa < sb) || (za == zb));
    
    int ea = pack_complex44(za);
    int eb = pack_complex44(zb);
    
    cout << "minimal_correlator_test(): start:"
	 << " f=" << f << ", sa=" << sa << ", sb=" << sb << ", t=" << t
	 << ", za=" << za << ", Ea=" << ea << ", zb=" << zb << ", Eb=" << eb
	 << endl;

    Correlator corr(nstations, nfreq);
    
    Array<int8_t> emat({nt_tot,nfreq,nstations}, af_zero);
    emat.at({t,f,sa}) = ea;
    emat.at({t,f,sb}) = eb;
    emat = emat.to_gpu();

    Array<int> vmat_cpu({nt_outer,nfreq,nstations,nstations,2}, af_zero);
    vmat_cpu.at({touter,f,sa,sa,0}) = (za * conj(za)).real();
    vmat_cpu.at({touter,f,sa,sb,0}) = (za * conj(zb)).real();
    vmat_cpu.at({touter,f,sa,sb,1}) = (za * conj(zb)).imag();
    vmat_cpu.at({touter,f,sb,sb,0}) = (zb * conj(zb)).real();
    
    Array<int> vmat_gpu({nt_outer,nfreq,nstations,nstations,2}, af_random | af_gpu);
    corr.launch(vmat_gpu, emat, nt_outer, nt_inner, nullptr, true);  // sync=true
    vmat_gpu = vmat_gpu.to_host();
    
    int nfail = 0;

    for (int to = 0; to < nt_outer; to++) {
	for (int f = 0; f < nfreq; f++) {
	    for (int i = 0; i < nstations; i++) {	    
		for (int k = i; k < nstations; k++) {
		    int cpu_re = vmat_cpu.at({to,f,i,k,0});
		    int cpu_im = vmat_cpu.at({to,f,i,k,1});
		    int gpu_re = vmat_gpu.at({to,f,i,k,0});
		    int gpu_im = vmat_gpu.at({to,f,i,k,1});

		    if ((cpu_re == 0) && (cpu_im == 0) && (gpu_re == 0) && (gpu_im == 0))
			continue;

		    bool fail = (cpu_re != gpu_re) || (cpu_im != gpu_im);

		    cout << "    " << (fail ? "FAILED" : "looks good")
			 << ": at touter=" << to << ", f=" << f << ", i=" << i << ", k=" << k << ": "
			 << "expected (re,im)=(" << cpu_re << "," << cpu_im << "), "
			 << "got (re,im)=(" << gpu_re << "," << gpu_im << ")" << endl;
		    
		    if (fail)
			nfail++;
		
		    if (nfail >= 32) {
			cout << "    Reached threshold failure count; aborting test" << endl;
			exit(1);
		    }
		}
	    }
	}
    }

    if (nfail > 1) {
	cout << "    Test failed" << endl;
	exit(1);
    }

    cout << "minimal_correlator_test(): pass" << endl;
}


void test_correlator(int nstations, int nfreq, int nt_outer, int nt_inner)
{
    const int nt_tot = (nt_outer * nt_inner);

    assert(!CorrelatorParams::artificially_remove_input_shuffle);
    assert(!CorrelatorParams::artificially_remove_output_shuffle);
    assert(!CorrelatorParams::artificially_remove_negate_4bit);

    cout << "\ntest_correlator("
	 << "nstations=" << nstations
	 << ", nfreq=" << nfreq
	 << ", nt_outer=" << nt_outer
	 << ", nt_inner=" << nt_inner
	 << ")" << endl;

    Array<int> vmat_cpu({nt_outer,nfreq,nstations,nstations,2}, af_zero);
    Array<int8_t> emat({nt_tot,nfreq,nstations}, af_zero);

    vector<int> ix(nstations);
    for (int i = 0; i < nstations; i++)
	ix[i] = i;

    constexpr int M = 10;
    vector<complex<int>> z(M);

    for (int touter = 0; touter < nt_outer; touter++) {
	for (int t = touter*nt_inner; t < (touter+1)*nt_inner; t++) {
	    for (int f = 0; f < nfreq; f++) {
		// Generate M random indices in [0:nstations).
		for (int i = 0; i < M; i++) {
		    int j = rand_int(i, nstations);
		    std::swap(ix[i], ix[j]);
		}
	    
		// Generate M random E-array values.
		for (int i = 0; i < M; i++) {
		    z[i] = { int(rand_int(-7,8)), int(rand_int(-7,8)) };
		    emat.at({t,f,ix[i]}) = pack_complex44(z[i]);
		}

		for (int i = 0; i < M; i++) {
		    for (int j = 0; j < M; j++) {
			complex<int> zz = z[i] * conj(z[j]);
			vmat_cpu.at({touter,f,ix[i],ix[j],0}) += zz.real();
			vmat_cpu.at({touter,f,ix[i],ix[j],1}) += zz.imag();
		    }
		}
	    }
	}
    }

    emat = emat.to_gpu();
    Array<int> vmat_gpu({nt_outer,nfreq,nstations,nstations,2}, af_random | af_gpu);
    
    Correlator corr(nstations, nfreq);
    corr.launch(vmat_gpu, emat, nt_outer, nt_inner, nullptr, true);  // sync=true
    vmat_gpu = vmat_gpu.to_host();

    for (int touter = 0; touter < nt_outer; touter++) {
	for (int f = 0; f < nfreq; f++) {
	    for (int i = 0; i < nstations; i++) {
		for (int k = i; k < nstations; k++) {
		    complex<int> vcpu = complex<int> (vmat_cpu.at({touter,f,i,k,0}), vmat_cpu.at({touter,f,i,k,1}));
		    complex<int> vgpu = complex<int> (vmat_gpu.at({touter,f,i,k,0}), vmat_gpu.at({touter,f,i,k,1}));
		    
		    if (vcpu != vgpu) {
			cout << "test_correlator() failed at touter=" << touter
			     << ", f=" << f << ", i=" << i << ", k=" << k
			     << ": vcpu = " << vcpu << ", vgpu = " << vgpu
			     << endl;
			
			exit(1);
		    }
		}
	    }
	}
    }
    
    cout << "test_correlator(): pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    test_pack_unpack();
    test_negate_4bit();
    test_transpose_rank8_4bit();

    // In the "minimal" test, only two entries in the E-array are nonzero, and
    // the test output is verbose. This can be useful for tracking down bugs,
    // but we don't run it by default.

    // (nstations, nfreq, f, sa, sb, t, za, ab)
    // minimal_correlator_test(128, 128, 0, 23, 37, 183, {1,2}, {3,4});  

    // Full end-to-end test starts here.
    
    std::mt19937 rng(137);
    const double maxbytes = 2.0e9;  // 2 GB

    // list of pairs (nstations, nfreq_max)
    vector<pair<int,int>> todo = {{128,1024},{1024,128}};
    
    for (auto p: todo) {
	int nstations = p.first;
	int nfreq_max = p.second;
	
	for (int nfreq = 1; nfreq <= nfreq_max; nfreq *= 2) {
	    double nbytes_e = nfreq * nstations;         // multiply by (nt_inner * nt_outer)
	    double nbytes_v = 8.0 * nfreq * nstations * nstations;  // multiply by (nt_outer)

	    int max_multiplier = int((0.9999*maxbytes - nbytes_v) / (256. * nbytes_e));
	    int nt_inner = 256 * gputils::rand_int(1, min(max_multiplier,10)+1, rng);

	    int max_nt_outer = int(maxbytes / (nt_inner*nbytes_e + nbytes_v));
	    int nt_outer = gputils::rand_int(1, max_nt_outer+1, rng);
	    test_correlator(nstations, nfreq, nt_outer, nt_inner);
	}
    }

    return 0;
}
