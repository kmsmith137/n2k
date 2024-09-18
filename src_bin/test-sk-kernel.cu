#include "../include/n2k/launch_s0_kernel.hpp"
#include "../include/n2k/internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>
#include <iostream>

using namespace std;
using namespace gputils;
using namespace n2k;


struct TestInstance
{
    long T = 0;
    long F = 0;
    long S = 0;
    long Nds = 0;

    double sk_rfimask_sigmas = 0.0;
    double single_feed_min_good_frac = 0.0;
    double feed_averaged_min_good_frac = 0.0;
    double mu_min = 0.0;
    double mu_max = 0.0;
    double rfi_mask_frac = 0.0;

    Array<float> out_sk_feed_averaged;    // shape (T,F,3)
    Array<float> out_sk_single_feed;      // shape (T,F,3,S)
    Array<uint> out_rfimask;              // shape (F,T*Nds/32)
    Array<uint> in_S012;                  // shape (T,F,3,S)
    Array<uint8_t> in_bf_mask;            // length S (bad feed bask)

    // Temp quantities used when generating the test instance.
    // All vectors are length-S.
    vector<uint> S0;
    vector<uint> S1;
    vector<uint> S2;
    vector<double> sf_sk;
    vector<double> sf_bias;
    vector<double> sf_sigma;
    double fsum_sk;
    double fsum_bias;
    double fsum_sigma;
    bool rfimask;
    
    
    TestInstance(long T_, long F_, long S_, long Nds_)
	: T(T_), F(F_), S(S_), Nds(Nds_)
    {
	assert(T > 0);
	assert(F > 0);
	assert(S > 0);
	assert(Nds > 0);
	assert((S % 128) == 0);
	assert((Nds % 32) == 0);
	
	this->out_sk_feed_averaged = Array<float> ({T,F,3}, af_rhost | af_zero);
	this->out_sk_single_feed = Array<float> ({T,F,3,S}, af_rhost | af_zero);
	this->out_rfimask = Array<uint> ({F,(T*Nds)/32}, af_rhost | af_zero);
	this->in_S012 = Array<uint> ({T,F,3,S}, af_rhost | af_zero);
	this->in_bf_mask = Array<uint8_t> ({S}, af_rhost | af_zero);

	this->sk_rfimask_sigmas = rand_uniform(0.5, 1.5);
	this->single_feed_min_good_frac = rand_uniform(0.7, 0.8);
	this->feed_averaged_min_good_frac = rand_uniform(0.3, 0.4);
	this->mu_min = rand_uniform(3.0, 4.0);
	this->mu_max = rand_uniform(20.0, 30.0);

	this->S0 = vector<uint> (S);
	this->S1 = vector<uint> (S);
	this->S2 = vector<uint> (S);
	this->sf_sk = vector<double> (S);
	this->sf_bias = vector<double> (S);
	this->sf_sigma = vector<double> (S);

	this->_init_bad_feed_mask();
	
	for (int t = 0; t < T; t++) {
	    for (int f = 0; f < F; f++) {
		this->_init_tf_pair();

		for (int s = 0; s < S; s++) {
		    this->in_S012.at({t,f,0,s}) = S0[s];
		    this->in_S012.at({t,f,1,s}) = S1[s];
		    this->in_S012.at({t,f,2,s}) = S2[s];
		    this->out_sk_single_feed.at({t,f,0,s}) = sf_sk[s];
		    this->out_sk_single_feed.at({t,f,1,s}) = sf_bias[s];
		    this->out_sk_single_feed.at({t,f,2,s}) = sf_sigma[s];
		}
		
		this->out_sk_feed_averaged.at({t,f,0}) = fsum_sk;
		this->out_sk_feed_averaged.at({t,f,1}) = fsum_bias;
		this->out_sk_feed_averaged.at({t,f,2}) = fsum_sigma;

		for (int i = t*(Nds/32); i < (t+1)*(Nds/32); i++)
		    this->out_rfimask.at({f,i}) = rfimask ? 0xffffffffU : 0;

		if (rfimask)
		    this->rfi_mask_frac += 1.0 / double(F*T);
	    }
	}
    }

    static inline double _compute_sk(double s0, double s1, double s2, double b)
    {
	double u = (s0 > 1.5) ? ((s0+1)/(s0-1)) : 0.0;
	double v = (s1 > 0.5) ? (s0 / (s1*s1)) : 0.0;
	return u * (v*s2 - 1) - b;
    }

    // Inverts (s2 -> sk) at fixed (s0,s1).
    static inline double _invert_sk(double s0, double s1, double sk, double b)
    {
	double ru = (s0 > 0.5) ? ((s0-1)/(s0+1)) : 0.0;
	double rv = (s0 > 0.5) ? ((s1*s1) / s0) : 0.0;
	return rv * (ru*(sk+b) + 1);
    }


    // Helper function called by constructor.
    void _init_bad_feed_mask()
    {
	// We mask up to 5% of the stations.
	
	for (int s = 0; s < S; s++)
	    in_bf_mask.at({s}) = 1;
	
	for (int i = 0; i < S/20; i++) {
	    int s = rand_int(0, S);
	    in_bf_mask.at({s}) = 0;
	}
    }


    // Helper function called by _init_tf_pair().
    void _init_valid_S0_S1(int s)
    {
	uint S0_edge = round(single_feed_min_good_frac * Nds);
	S0[s] = rand_int(S0_edge+1, Nds+1);
	
	uint S1_edge0 = round(mu_min * S0[s]);
	uint S1_edge1 = round(mu_max * S0[s]);
	S1[s] = rand_int(S1_edge0+1, S1_edge1);
    }


    // Helper function called by _init_tf_pair().
    void _init_invalid_S0_S1(int s)
    {
	for (;;) {
	    S0[s] = rand_int(-Nds/32, Nds+1);
	    S0[s] = max(S0[s], 0);
	    S1[s] = rand_int(0, 98*S0[s]+1);
	    
	    uint S0_edge = round(single_feed_min_good_frac * Nds);
	    uint S1_edge0 = round(mu_min * S0[s]);
	    uint S1_edge1 = round(mu_max * S0[s]);

	    if ((S0[s] < S0_edge) || (S1[s] < S1_edge0) || (S1[s] > S1_edge1))
		return;
	}
    }
    
    void _init_tf_pair()
    {
	double p1 = rand_uniform(-0.2, 1.0);
	double p2 = rand_uniform(-0.2, 1.0);
	double prob_sf_valid = max(max(p1,p2), 0.0);
	
	// The purpose of this outer loop is to allow restarts, if we end up in
	// a situation where roundoff error may be an issue for the unit test
	// (because we're close to a boolean threshold).
	
	for (;;) {
	    double sum_w = 0.0;
	    double sum_wsk = 0.0;
	    double sum_wb = 0.0;
	    double sum_wsigma2 = 0.0;
	    
	    for (int s = 0; s < S; s++) {
		bool sf_valid = (rand_uniform() < prob_sf_valid);

		// Init S0[s], S1[s].
		if (sf_valid)
		    _init_valid_S0_S1(s);
		else
		    _init_invalid_S0_S1(s);

		// Code after this point initializes S2[s].

		double s0 = S0[s];
		double s1 = S1[s];
		double mu = (s0 > 0.5) ? (s1/s0) : 0.0;
		double b = 0.001 * mu;                          // FIXME placeholder for testing
		double sigma = (s0 > 0.5) ? sqrt(4/s0) : -1.0;  // FIXME placeholder for testing
		double target_sk = 1.0 + sigma * sqrt(3.) * rand_uniform(-1.0,1.0);
		double s2 = _invert_sk(s0, s1, target_sk, b);

		s2 = max(s2, s1);
		s2 = min(s2, 98*s1);		
		s2 = round(s2);
		S2[s] = s2;
		
		double actual_sk = _compute_sk(s0, s1, s2, b);

		sf_sk[s] = sf_valid ? actual_sk : 0.0;
		sf_bias[s] = sf_valid ? b : 0.0;
		sf_sigma[s] = sf_valid ? sigma : -1.0;
		
		double w = (sf_valid && in_bf_mask.at({s})) ? s0 : 0.0;
		
		sum_w += w;
		sum_wsk += w * sf_sk[s];
		sum_wb += w * sf_bias[s];
		sum_wsigma2 += w * w * sf_sigma[s] * sf_sigma[s];
	    }

	    double sum_w_threshold = feed_averaged_min_good_frac * S * Nds;

	    if (fabs(sum_w - sum_w_threshold) < 0.1)
		continue;   // Restart (too close to boolean threshold)
	    
	    bool fsum_valid = (sum_w > sum_w_threshold);

	    this->fsum_sk = fsum_valid ? (sum_wsk / sum_w) : 0.0;
	    this->fsum_bias = fsum_valid ? (sum_wb / sum_w) : 0.0;
	    this->fsum_sigma = fsum_valid ? (sqrt(sum_wsigma2) / sum_w) : -1.0;

	    if (!fsum_valid) {
		this->rfimask = 0;
		return;
	    }

	    // RFI mask is determined by thresholding u.
	    double u = fabs(fsum_sk - 1.0);
	    double uthresh = sk_rfimask_sigmas * fsum_sigma;
	    assert(uthresh > 2.0e-4);

	    if (fabs(u-uthresh) < 1.0e-4)
		continue;  // Restart (too close to boolean threshold)
	    
	    this->rfimask = (u < uthresh);
	    return;
	}
    }
};


static void test_sk_kernel(const TestInstance &ti, bool check_sf_sk=true, bool check_rfimask=true)
{    
    long T = ti.T;
    long F = ti.F;
    long S = ti.S;
    long Nds = ti.Nds;
    
    cout << "test_sk_kernel: T=" << T << ", F=" << F << ", S=" << S << ", Nds=" << Nds
	 << ", check_sf_sk=" << check_sf_sk << ", check_rfimask=" << check_rfimask
	 << ", rfi_mask_frac=" << ti.rfi_mask_frac << endl;
    
    // Input arrays
    Array<uint> gpu_S012 = ti.in_S012.to_gpu();
    Array<uint8_t> gpu_bf_mask = ti.in_bf_mask.to_gpu();

    // Output arrays
    // FIXME test rfimask_fstride.
    Array<float> gpu_sk_feed_averaged({T,F,3}, af_gpu | af_random);
    Array<float> gpu_sk_single_feed;
    Array<uint> gpu_rfimask;

    if (check_sf_sk)
	gpu_sk_single_feed = Array<float> ({T,F,3,S}, af_gpu | af_random);
    if (check_rfimask)
	gpu_rfimask = Array<uint> ({F,(T*Nds)/32}, af_gpu | af_random);
    
    launch_sk_kernel(
        gpu_sk_feed_averaged,
	gpu_sk_single_feed,
	gpu_rfimask,
	gpu_S012,
	gpu_bf_mask,
	ti.sk_rfimask_sigmas,
	ti.single_feed_min_good_frac,
	ti.feed_averaged_min_good_frac,
	ti.mu_min,
	ti.mu_max,
	ti.Nds);

    CUDA_CALL(cudaDeviceSynchronize());

    if (check_sf_sk)
	gputils::assert_arrays_equal(gpu_sk_single_feed, ti.out_sk_single_feed, "gpu_sf_sk", "ref_sf_sk", {"t","f","n","s"});
	
    gputils::assert_arrays_equal(gpu_sk_feed_averaged, ti.out_sk_feed_averaged, "gpu_fsum_sk", "ref_fsum_sk", {"t","f","n"});

    if (check_rfimask)
	gputils::assert_arrays_equal(gpu_rfimask, ti.out_rfimask, "gpu_rfimask", "ref_rfimask", {"f","t32"});
}


static void test_sk_kernel()
{
    long T = rand_int(1, 21);
    long F = rand_int(1, 21);
    long S = 128 * rand_int(1, 17);
    long Nds = 32 * rand_int(1, 11);
    bool check_sf_sk = (rand_uniform() < 0.9);
    bool check_rfimask = (rand_uniform() < 0.9);

    TestInstance ti(T,F,S,Nds);
    test_sk_kernel(ti, check_sf_sk, check_rfimask);
}


int main(int argc, char **argv)
{
    for (int i = 0; i < 100; i++)
	test_sk_kernel();
    
    return 0;
}
