#include <iostream>
#include <gputils.hpp>
#include "n2k.hpp"

using namespace std;
using namespace gputils;
using namespace n2k;

constexpr int default_nt_inner = 256 * 1024;
constexpr int default_nt_tot = 256 * 1024;
constexpr int default_nfreq = 16;
constexpr int default_nstreams = 1;
constexpr int default_ncallbacks = 300;
constexpr int max_nt_outer = 64;


static void time_correlator(int nfreq, int nt_outer, int nt_inner, int nstreams, int ncallbacks)
{
    cout << "time-correlator:"
	 << " nt_inner=" << nt_inner
	 << ", nt_outer=" << nt_outer
	 << ", nfreq=" << nfreq
	 << ", nstreams=" << nstreams
	 << ", ncallbacks=" << ncallbacks
	 << endl;

    ssize_t nstat = 1024; // XXX
    ssize_t nt_tot = nt_outer * nt_inner;
    
    double varr_gb = nstreams * nt_outer * nfreq * pow(nstat,2.) * 2. / pow(2,30.);
    double earr_gb = nstreams * nt_tot * double(nfreq*nstat) / pow(2,30);
    double rt_sec = (nfreq/16.) * (nt_tot * 1.707e-6);  // CHORD: 16 freqs per gpu, 1.707 usec samples
    double vsample_sec = nt_inner * 1.707e-6;

    cout << "GPU memory usage will be " << (varr_gb + earr_gb) << " GB\n"
	 << "This data volume corresponds to " << rt_sec << " seconds of real-time CHORD data\n"
	 << "Visibility matrix sampling rate: " << vsample_sec << " sec in CHORD\n"
	 << endl;
    
    Correlator corr(nstat, nfreq);
    vector<Array<int>> varr(nstreams);
    vector<Array<int8_t>> earr(nstreams);

    for (int i = 0; i < nstreams; i++) {
	varr[i] = Array<int> ({nt_outer, nfreq, nstat, nstat, 2}, af_zero | af_gpu);
	earr[i] = Array<int8_t> ({nt_tot, nfreq, nstat}, af_zero | af_gpu);
    }

    cout << "The first few kernels run slow, for reasons I haven't understood yet!\n"
	 << "To mitigate this, the first kernel will be untimed.\n" << endl;
    
    corr.launch(varr[0], earr[0], nt_outer, nt_inner, nullptr, true);

    cout << "Now running timed kernels. The first few kernels will be a few percent\n"
	 << "slower than the long-run average, but the timing will quickly settle down.\n"
	 << endl;

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
	{
	    corr.launch(varr[istream], earr[istream], nt_outer, nt_inner, stream);
	};
    
    stringstream sp_name;
    sp_name << "n2k (vsamp=" << vsample_sec << ")";
    
    CudaStreamPool sp(callback, ncallbacks, nstreams, sp_name.str());
    sp.monitor_time("real-time fraction", rt_sec);
    sp.run();

    // Wrapper script 'run.sh' will capture this last line with 'tail -1'.
    cout << vsample_sec << " " << (sp.time_per_callback / rt_sec) << endl;
}


static void usage(bool cond=false)
{
    if (cond)
	return;
    
    cerr << "usage: time-correlator [nt_inner] [nt_outer] [nfreq] [nstreams] [ncallbacks]\n"
	 << "    default nt_inner = " << default_nt_inner << "\n"
	 << "    default nt_outer = min(" << max_nt_outer << ", " << default_nt_tot << "/nt_inner)\n"
	 << "    default nfreq = " << default_nfreq << "\n"
	 << "    default nstreams = " << default_nstreams << "\n"
	 << "    default ncallbacks = " << default_ncallbacks << "\n"
	 << endl;

    exit(1);
}


int main(int argc, const char **argv)
{
    usage(argc <= 6);

    int nt_inner = (argc >= 2) ? gputils::from_str<int>(argv[1]) : default_nt_inner;
    int nt_outer = (argc >= 3) ? gputils::from_str<int>(argv[2]) : min(max_nt_outer, (default_nt_tot+nt_inner-1)/nt_inner);
    int nfreq = (argc >= 4) ? gputils::from_str<int>(argv[3]) : default_nfreq;
    int nstreams = (argc >= 5) ? gputils::from_str<int>(argv[4]) : default_nstreams;
    int ncallbacks = (argc >= 6) ? gputils::from_str<int>(argv[5]) : default_ncallbacks;
	
    time_correlator(nfreq, nt_outer, nt_inner, nstreams, ncallbacks);
    return 0;
}
