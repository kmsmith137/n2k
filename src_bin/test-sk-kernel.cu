#include "../include/n2k/launch_s0_kernel.hpp"
#include "../include/n2k/internals.hpp"

#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>
#include <iostream>

using namespace std;
using namespace gputils;
using namespace n2k;

// TODO make sure rfimask_fstride is tested.


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

    Array<float> out_sk_feed_averaged;    // shape (T,F,3)
    Array<float> out_sk_single_feed ;     // shape (T,F,3,S)
    Array<uint> out_rfimask;              // shape (F,T*Nds/32)
    Array<uint> in_S012;                  // shape (T,F,3,S)
    Array<uint8_t> in_bf_mask;            // length S (bad feed bask)

    // Temp quantities used when generating the test instance.
    // All vectors are length-S.
    vector<uint> S0b;
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
	this->feed_averaged_good_frac = rand_uniform(0.3, 0.4);
	this->mu_min = rand_uniform(3.0, 4.0);
	this->mu_max = rand_uniform(20.0, 30.0);

	this->S0 = vector<uint> (S);
	this->S1 = vector<uint> (S);
	this->S2 = vector<uint> (S);
	this->S0b = vector<uint> (S);

	this->_init_bad_feed_mask();
	
	for (int t = 0; t < T; t++) {
	    for (int f = 0; f < F; f++) {
		this->_init_S0b();
		this->_init_S0_S1();
		this->_init_S2();  // also initializes output statistics (SK, rfimask).

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

		for (int i = t*(Tds/32); i < (t+1)*(Tds/32); i++)
		    this->out_rfimask.at({f,i}) = rfimask ? 0xffffffffU : 0;
	    }
	}
    }


    inline bool _is_valid(double s0, double s1)
    {
	double mu = (s0 > 0.5) ? (s1/s0) : 0.0;
	bool s0_valid = (s0 >= single_feed_min_good_frac * Tds);
	bool s1_valid = (mu >= mu_min) && (mu <= mu_max);
	return s0_valid && s1_valid;
    }
    
    inline double _compute_bias(double s0, double s1)
    {
	// FIXME placeholder for testing
	double mu = (s0 > 0.5) ? (s0/s1) : 0;
	return 0.01 * mu;
    }
    
    inline double _compute_sigma(double s0)
    {
	// FIXME placeholder for testing
	return (s0 > 0.5) ? sqrt(4/s0) : -1.0;
    }

    inline double _compute_sk(double s0, double s1, double s2, double b)
    {
	double u = (s0 > 1.5) ? ((s0+1)/(s0-1)) : 0.0;
	double v = (s1 > 0.5) ? (s0 / (s1*s1)) : 0.0;
	return u * (v*s2 - 1) - b;
    }

    // Inverts (s2 -> sk) at fixed (s0,s1).
    inline double _invert_sk(double s0, double s1, double sk, double b)
    {
	double ru = (s0 > 0.5) ? ((s0-1)/(s0+1)) : 0.0;
	double rv = (s0 > 0.5) ? ((s1*s1) / s0) : 0.0;
	return rv * (ru*(s2+b) + 1);
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

    
    // Helper function called by constructor.
    void _init_S0b()
    {
	for (int s = 0; s < S; s++)
	    S0b[s] = 0;

	// Test corner case where all single-feed SK statistics are invalid.
	if (rand_uniform() < 0.05)
	    return;

	int nedge = round(feed_averaged_min_good_frac * S * Tds);
	bool is_above_edge = (rand_uniform() < 0.8);
	int n = 0;

	for (;;) {
	    int m = rand_int(S0b_min, Tds+1);
	    int s = rand_int(0,S);

	    if (is_above_edge && (n > nedge))
		return;	    
	    if (!is_above_edge && ((n+m <= nedge)))
		return;
	    if (S0b[s] || !in_bf_mask.at({s}))
		continue;

	    S0b[s] = m;
	    n += m;
	}
    }

    // Helper function called by constructor.
    void _init_S0_S1()
    {
	int S0_edge = round(feed_averaged_good_frac * Tds);
	    
	for (int s = 0; s < S; s++) {
	    S0[s] = (S0b[s] > 0) ? S0b[s] : rand_int(0,Tds+1);

	    if (S0[s] == 0) {
		S1[s] = 0;
		continue;
	    }	

	    int S1_edge0 = round(mu_min * S0[s]);
	    int S1_edge1 = round(mu_max * S0[s]);
	    int S1_max = min(S1_edge1 + 10*S0[s], 98*S0[s]);

	    if (S0b[s] > 0)
		S1[s] = rand_int(S1_edge0+1, S1_edge1);   // S1 succeeds
	    else if (!in_bf.mask.at({s}) || (S0[s] < S0_edge))
		S1[s] = rand_int(0, S1_max+1);            // no constraint on S1
	    else if (rand_uniform() < 0.5)
		S1[s] = rand_int(0, S1_edge0);            // S1 fails on low side
	    else
		S1[s] = rand_int(S1_edge1+1, S1_max+1);   // S1 fails on high side
	}
    }

    // Helper function called by constructor.
    // Also initializes output arrays (SK, rfimask).
    
    void _init_S2()
    {
	for (;;) {
	    double sum_w = 0.0;
	    double sum_wsk = 0.0;
	    double sum_wb = 0.0;
	    double sum_wsigma2 = 0.0;

	    for (int s = 0; s < S; s++) {
		if ((S0[s] <= 1) || (S1[s] == 0)) {
		    S2[s] = S1[s]*S1[s];
		    continue;
		}

		double s0 = S0[s];
		double s1 = S1[s];

		double b = _compute_bias(s0, s1);
		double sigma = _compute_sigma(s0);
		double target_sk = 1.0 + sigma * sqrt(3.) * rand_uniform(-1.0,1.0);
		double s2 = _invert_sk(s0, s1, sk, b);
		
		s2 = max(s2, s1);
		s2 = min(s2, 98*s1);
		s2 = round(s2);
		S2[s] = s2;

		double actual_sk = _compute_sk(s0, s1, s2, b);
		bool sf_valid = _is_valid(s0, s1);

		sf_sk[s] = sf_valid ? actual_sk : 0.0;
		sf_bias[s] = sf_valid ? b : 0.0;
		sf_sigma[s] = sf_valid ? sigma : -1.0;
		
		double w = (sf_valid && in_bf_mask.at({s})) ? s0 : 0.0;
		
		sum_w += w;
		sum_wsk += w * sf_sk[s];
		sum_wb += w * sf_bias[s];
		sum_wsigma2 += w * w * sf_sigma[s] * sf_sigma[s];
	    }

	    bool fsum_valid = (sum_w > (feed_averaged_min_good_frac * S * Nds));

	    this->fsum_sk = fsum_valid ? (sum_wsk / sum_w) : 0.0;
	    this->fsum_bias = fsum_valid ? (sum_wb / sum_w) : 0.0;
	    this->fsum_sigma = fsum_valid ? (sqrt(sum_wsigma2) / sum_w) : -1.0;

	    if (!fsum_valid) {
		this->rfimask = 0;
		return;
	    }

	    if (out_rfimask.data != NULL) {
		bool rfi_good = fsum_valid
		    && (fsum_sk >= 1.0f - sk_rfimask_sigmas * fsum_sigma)
		    && (fsum_sk <= 1.0f + sk_rfimask_sigmas * fsum_sigma);
		
		for (int i = t*(Tds/32); i < (t+1)*(Tds/32); i++)
		    out_rfimask.at({f,i}) = rfi_good ? 0xffffffffU : 0;
	    }

	bool 
	// Close loophole where feed-averaged SK is too close to threshold,
	// making the 
	
	if (sum_w 
	// Close loophole where .
	
	if (sum_w < min_ *sum_w)
	    return;

	double sk = sum_wsk / sum_w;
	double sigma = sqrt(sum_wsigma2) / sum_w;
	
	if ()
	    continue;
    }
}

    
static void reference_sk_kernel(
    Array<float> &out_sk_feed_averaged,   // shape (T,F,3)
    Array<float> &out_sk_single_feed,     // either empty array or shape (T,F,3,S)
    Array<uint> &out_rfimask,             // either empty array or shape (F,T*Nds/32), need not be contiguous
    const Array<uint> &in_S012,           // shape (T,F,3,S)
    const Array<uint8_t> &in_bf_mask,     // length S (bad feed bask)
    double sk_rfimask_sigmas,
    double single_feed_min_good_frac,     // See comments in sk_kernel.hpp
    double feed_averaged_min_good_frac,   // See comments in sk_kernel.hpp
    double mu_min,                         // See comments in sk_kernel.hpp
    double mu_max,                         // See comments in sk_kernel.hpp
    long Nds)                             // S012 downsampling factor
{
    // No argument checking -- assumes launch_sk_kernel() has already been called.
    
    int T = in_S012.shape[0];
    int F = in_S012.shape[1];
    int S = in_S012.shape[3];

    for (int t = 0; t < T; t++) {
	for (int f = 0; f < F; f++) {
	    double sum_w = 0.0;
	    double sum_wsk = 0.0;
	    double sum_wb = 0.0;
	    double sum_wsigma2 = 0.0;
	    
	    for (int s = 0; s < S; s++) {
		double s0 = in_S012.at({t,f,0,s});
		double s1 = in_S012.at({t,f,1,s});
		double s2 = in_S012.at({t,f,2,s});

		bool sf_valid = _is_valid(s0, s1);
		double sf_b = sf_valid ? _compute_bias(s0, s1) : 0.0;
		double sf_sk = sf_valid ? _compute_sk(s0, s1, s2, sf_b) : 0.0;
		double sf_sigma = sf_valid ? _compute_sigma(s0) : -1.0;
		double w = (sf_valid && in_bf_mask.at({s})) ? s0 : 0.0;
		
		sum_w += w;
		sum_wsk += w * sf_sk;
		sum_wb += w * sf_b;
		sum_wsigma2 += w * w * sf_sigma * sf_sigma;

		if (out_sk_single_feed.data) {
		    out_sk_single_feed.at({t,f,0,s}) = sf_sk;
		    out_sk_single_feed.at({t,f,0,s}) = sf_b;
		    out_sk_single_feed.at({t,f,0,s}) = sf_sigma;
		}
	    }

	    double fsum_good_frac = sum_w / double(T*Nds);
	    bool fsum_valid = (fsum_good_frac >= feed_averaged_min_good_frac);

	    if (fsum_valid)
		assert(sum_w > 0.5);

	    double fsum_sk = fsum_valid ? (sum_wsk / sum_w) : 0.0;
	    double fsum_b = fsum_valid ? (sum_wb / sum_w) : 0.0;
	    double fsum_sigma = fsum_valid ? (sqrt(sum_wsigma2) / sum_w) : -1.0;

	    out_sf_feed_averaged.at({t,f,0}) = fsum_sk;
	    out_sf_feed_averaged.at({t,f,1}) = fsum_b;
	    out_sf_feed_averaged.at({t,f,2}) = fsum_sigma;

	    if (out_rfimask.data != NULL) {
		bool rfi_good = fsum_valid
		    && (fsum_sk >= 1.0f - sk_rfimask_sigmas * fsum_sigma)
		    && (fsum_sk <= 1.0f + sk_rfimask_sigmas * fsum_sigma);
		
		for (int i = t*(Tds/32); i < (t+1)*(Tds/32); i++)
		    out_rfimask.at({f,i}) = rfi_good ? 0xffffffffU : 0;
	    }
	}
    }
}


static void run_test(long T, long F, long S, long Nds, bool have_sf_sk, bool have_rfimask)
{
    TestInstance ti(T, F, S, Nds, have_sf_sk, have_rfimask);

    Array<float> gpu_sk_feed_averaged = ti.out_sk_feed_averaged.to_gpu();
    Array<uint> gpu_S012 = ti.in_S012.to_gpu();
    Array<uint8_t> gpu_bf_mask = ti.in_bf_mask.to_gpu();
    Array<float> gpu_sk_single_feed;
    Array<uint> gpu_rfimask;

    if (have_sf_sk)
	gpu_sk_single_feed = ti.out_sk_single_feed.to_gpu();
    if (have_rfimask)
	gpu_rfimask = ti.out_rfimask.to_gpu();
    
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
	
    Array<float> &out_sk_feed_averaged,   // shape (T,F,3)
    Array<float> &out_sk_single_feed,     // either empty array or shape (T,F,3,S)
    Array<uint> &out_rfimask,             // either empty array or shape (F,T*Nds/32), need not be contiguous
    const Array<uint> &in_S012,           // shape (T,F,3,S)
    const Array<uint8_t> &in_bf_mask,     // length S (bad feed bask)
    double sk_rfimask_sigmas,             // RFI masking threshold in "sigmas" (only used if out_rfimask != NULL)
    double single_feed_min_good_frac,     // See comments in sk_kernel.hpp
    double feed_averaged_min_good_frac,   // See comments in sk_kernel.hpp
    double mu_min,                         // See comments in sk_kernel.hpp
    double mu_max,                         // See comments in sk_kernel.hpp
    long Nds,                             // S012 downsampling factor
    launch_
}

int main(int argc, char **argv)
{
    
    return 0;
}
