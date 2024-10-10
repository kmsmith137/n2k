#include "../include/n2k/rfi_kernels.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace n2k;
using namespace gputils;


// FIXME move somewhere more general?
inline int quantize(double x)
{
    x = std::abs(x);
    int i = int(x + 0.5);
    return std::min(i,7);   // Note: we drop the sign!
}


struct SkTracker
{
    double n = 0;
    double sum_x = 0;
    double sum_x2 = 0;
    double predicted_sum_x2 = 0;
    
    void update(float sk, float sigma)
    {
	if (sigma > 0) {
	    double x = (sk-1.0);
	    n += 1.0;
	    sum_x += x;
	    sum_x2 += x*x;
	    predicted_sum_x2 += sigma*sigma;
	}
    }

    void show(const string &name)
    {
	if (n < 0.5) {
	    cout << name << ": no valid sk-samples" << endl;
	    return;
	}
	
	cout << "    " << name
	     << ": <SK-1> = " << (sum_x / n)
	     << " (single sk-array element bias is " << (sum_x / sqrt(sum_x2*n)) << " sigma,"
	     << " cumulative bias across all MCs is " << (sum_x / sqrt(predicted_sum_x2)) << " sigma)."
	     << " Ratio (rms/rms_predicted) = " << sqrt(sum_x2 / predicted_sum_x2) << endl;
    }
};


// -------------------------------------------------------------------------------------------------


struct RunState
{
    // Caller must initialize these members before calling init().
    string name;
    long T = 0;
    long F = 0;
    long S = 0;
    long Nds1 = 0;      // Baseband -> 1 ms
    long Nds2 = 0;      // 1 ms -> 30 ms
    double rms = 0.0;
    bool offset_encoded = true;

    // Subsequent members are not initialized by caller.
    long nframes = 0;
    shared_ptr<SkKernel> sk_kernel1;
    shared_ptr<SkKernel> sk_kernel2;

    // GPU arrays.
    Array<uint8_t> E;        // shape (T,F,S)
    Array<uint8_t> bf_mask;  // shape (S,)
    Array<ulong> pl_mask;    // shape (T/128, (F+3)/4, S/8)
    Array<ulong> S012_fast;  // shape (T/Nds1,F,3,S)
    Array<ulong> S012_slow;  // shape (T/(Nds1*Nds2),F,3,S)
    Array<uint> rfimask;     // shape (F,T/32)
    Array<float> sk_feed_averaged_fast;   // shape (T/Nds1,F,3)
    Array<float> sk_feed_averaged_slow;   // shape (T/(Nds1*Nds2),F,3)
    Array<float> sk_single_feed_slow;     // shape (T/(Nds1*Nds2),F,3,S)

    // CPU arrays.
    Array<uint8_t> E_cpu;
    Array<uint8_t> bf_mask_cpu;
    Array<ulong> pl_mask_cpu;
    Array<uint>  rfimask_cpu;
    Array<float> sk_feed_averaged_fast_cpu;
    Array<float> sk_feed_averaged_slow_cpu;
    Array<float> sk_single_feed_slow_cpu;

    // Track SK-statistics.
    SkTracker sk_feed_averaged_fast_tracker;
    SkTracker sk_feed_averaged_slow_tracker;
    SkTracker sk_single_feed_slow_tracker;

    // Track RFI mask fraction.
    double rfimask_num = 0.0;
    double rfimask_den = 0.0;

    
    void init()
    {
	// FIXME move these to main() for visibility?
	SkKernel::Params p;
	p.sk_rfimask_sigmas = 1.5;
	p.single_feed_min_good_frac = 0.5;
	p.feed_averaged_min_good_frac = 0.5;
	p.mu_min = 1.0;
	p.mu_max = 50.0;

	// sk_kernel1
	p.Nds = Nds1;
	this->sk_kernel1 = make_shared<SkKernel> (p);

	// sk_kernel2
	p.Nds = Nds1 * Nds2;
	this->sk_kernel2 = make_shared<SkKernel> (p);

	// GPU arrays.
	this->E = Array<uint8_t> ({T,F,S}, af_gpu);
	this->bf_mask = Array<uint8_t> ({S}, af_gpu);
	this->pl_mask = Array<ulong> ({T/128,(F+3)/4,S/8}, af_gpu);
	this->S012_fast = Array<ulong> ({T/Nds1,F,3,S}, af_gpu);
	this->S012_slow = Array<ulong> ({T/(Nds1*Nds2),F,3,S}, af_gpu);
	this->rfimask = Array<uint> ({F,T/32}, af_gpu);
	this->sk_feed_averaged_fast = Array<float> ({T/Nds1,F,3}, af_gpu);
	this->sk_feed_averaged_slow = Array<float> ({T/(Nds1*Nds2),F,3}, af_gpu);
	this->sk_single_feed_slow = Array<float> ({T/(Nds1*Nds2),F,3,S}, af_gpu);

	// CPU arrays.
	this->E_cpu = Array<uint8_t> ({T,F,S}, af_rhost);
	this->bf_mask_cpu = Array<uint8_t> ({S}, af_rhost);
	this->pl_mask_cpu = Array<ulong> ({T/128,(F+3)/4,S/8}, af_rhost);
	this->rfimask_cpu = Array<uint> ({F,T/32}, af_rhost);
	this->sk_feed_averaged_fast_cpu = Array<float> ({T/Nds1,F,3}, af_rhost);
	this->sk_feed_averaged_slow_cpu = Array<float> ({T/(Nds1*Nds2),F,3}, af_rhost);
	this->sk_single_feed_slow_cpu = Array<float> ({T/(Nds1*Nds2),F,3,S}, af_rhost);
    }

    
    void run_frame()
    {
	// No masks for now (might put these in later but I don't think it's a high priority)
	memset(bf_mask_cpu.data, 0xff, bf_mask_cpu.size);
	memset(pl_mask_cpu.data, 0xff, pl_mask_cpu.size * sizeof(ulong));
	
	int8_t bits = offset_encoded ? 0x88 : 0;
	std::normal_distribution<double> dist(0, 1.0);

	for (long t = 0; t < T; t++) {
	    for (long f = 0; f < F; f++) {
		for (long s = 0; s < S; s++) {
		    int ex = quantize(rms * dist(gputils::default_rng));
		    int ey = quantize(rms * dist(gputils::default_rng));
		    int8_t e44 = ((ex & 0xf) | ((ey & 0xf) << 4)) ^ bits;
		    E_cpu.data[t*F*S + f*S + s] = e44;
		}
	    }
	}

	// Copy CPU -> GPU.
	E.fill(E_cpu);
	bf_mask.fill(bf_mask_cpu);
	pl_mask.fill(pl_mask_cpu);

	// Launch kernels.
	
	Array<ulong> S0 = S012_fast.slice(2,0);
	Array<ulong> S12 = S012_fast.slice(2,1,3);
	Array<float> empty_sk_single_feed;
	Array<uint> empty_rfimask;

	launch_s0_kernel(S0, pl_mask, Nds1);
	launch_s12_kernel(S12, E, Nds1, offset_encoded);
	launch_s012_time_downsample_kernel(S012_slow, S012_fast, Nds2);
	sk_kernel1->launch(sk_feed_averaged_fast, empty_sk_single_feed, rfimask, S012_fast, bf_mask);
	sk_kernel2->launch(sk_feed_averaged_slow, sk_single_feed_slow, empty_rfimask, S012_slow, bf_mask);
	CUDA_CALL(cudaDeviceSynchronize());
    
	// Copy GPU -> CPU.
	rfimask_cpu.fill(rfimask);
	sk_feed_averaged_fast_cpu.fill(sk_feed_averaged_fast);
	sk_feed_averaged_slow_cpu.fill(sk_feed_averaged_slow);
	sk_single_feed_slow_cpu.fill(sk_single_feed_slow);

	// Update SKTrackers
	
	for (long t = 0; t < T/Nds1; t++) {
	    for (long f = 0; f < F; f++) {
		float sk = sk_feed_averaged_fast_cpu.data[t*(3*F) + 3*f];
		float sigma = sk_feed_averaged_fast_cpu.data[t*(3*F) + 3*f + 2];
		sk_feed_averaged_fast_tracker.update(sk, sigma);
	    }
	}

	for (long t = 0; t < T/(Nds1*Nds2); t++) {
	    for (long f = 0; f < F; f++) {
		float sk = sk_feed_averaged_slow_cpu.data[t*(3*F) + 3*f];
		float sigma = sk_feed_averaged_slow_cpu.data[t*(3*F) + 3*f + 2];
		sk_feed_averaged_slow_tracker.update(sk, sigma);

		for (long s = 0; s < S; s++) {
		    sk = sk_single_feed_slow_cpu.data[t*(3*F*S) + f*(3*S) + s];
		    sigma = sk_single_feed_slow_cpu.data[t*(3*F*S) + f*(3*S) + 2*S + s];
		    sk_single_feed_slow_tracker.update(sk, sigma);
		}
	    }
	}

	// Update RFI mask fraction tracking.
	
	for (long f = 0; f < F; f++) {
	    for (long t32 = 0; t32 < T/32; t32++) {
		uint m = rfimask_cpu.data[f*(T/32) + t32];
		assert((m == 0) || (m == 0xffffffffU));
		rfimask_num += (m ? 1.0 : 0.0);
		rfimask_den += 1.0;
	    }
	}

	this->nframes++;
    }


    void show_statistics()
    {
	cout << name << ": nframes=" << nframes << endl;
	sk_feed_averaged_fast_tracker.show("Feed-averaged fast SK");
	sk_feed_averaged_slow_tracker.show("Feed-averaged slow SK");
	sk_single_feed_slow_tracker.show("Single-feed slow SK");

	if (rfimask_den > 0.0)
	    cout << "    RFI mask fraction = " << (rfimask_num / rfimask_den) << endl;
    }
};



int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "Usage: test-sk-bias <rms>" << endl;
	return 2;
    }

    RunState rs;
    rs.Nds1 = 128;
    rs.Nds2 = 48;
    rs.T = rs.Nds1 * rs.Nds2 * 4;
    rs.F = 51;
    rs.S = 1024;
    rs.rms = gputils::from_str<double> (argv[1]);
    rs.offset_encoded = true;
    
    stringstream ss;
    ss << "Full CHORD (rms=" << rs.rms << ")";
    rs.name = ss.str();

    rs.init();
    
    cout << "Starting Monte Carlos.\n"
	 << "Warning: slow! (Simulating E-array on CPU is the bottleneck).\n"
	 << "You may want to leave this running in the background, and check on it later." << endl;

    for (;;) {
	rs.run_frame();
	rs.show_statistics();
    }
}
