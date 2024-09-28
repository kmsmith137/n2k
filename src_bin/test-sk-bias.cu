#include "../include/n2k.hpp"
#include "../include/n2k/interpolation.hpp"

#include <gputils/string_utils.hpp>
#include <iostream>

using namespace std;
using namespace n2k;


static bool is_perfect_square(int n)
{
    if (n < 1)
	return false;
    int m = int(sqrt(n) + 0.5);
    return n == m*m;
}


// FIXME move to libn2k.so (needs GPU-friendly interface)
template<typename T>
struct Bvec
{
    int n = 0;
    double *bias_coeffs = nullptr;
    
    int min_s1 = 0;
    int max_s1 = 0;

    // Length 98*n+1
    vector<T> bvec;

    // Helper function for constructor.
    // No range check on y!
    inline double _eval_y(int ix, double y)
    {
	using sk_globals::nx;
	using sk_globals::ny;

	assert((ix >= 0) && (ix < nx));
	
	double ret = 0.0;
	double ypow = 1.0;
	
	for (int j = 0; j < ny; j++) {
	    ret += bias_coeffs[ix*ny+j] * ypow;
	    ypow *= y;
	}

	return ret;
    }
	
    
    // Helper function for constructor.
    inline double _interpolate(double x, double y)
    {
	using sk_globals::xmin;
	using sk_globals::xmax;
	using sk_globals::nx;

	// Normalize to [0,nx-1]
	double t = (nx-1) * (x-xmin) / (xmax-xmin);
	assert(t >= 0.9);
	assert(t <= nx-1.9);
	
	int it = int(t);
	it = std::max(it,1);
	it = std::min(it,nx-3);

	double b0 = _eval_y(it-1, y);
	double b1 = _eval_y(it, y);
	double b2 = _eval_y(it+1, y);
	double b3 = _eval_y(it+2, y);

	return cubic_interpolate(t-it, b0, b1, b2, b3);
    }
    
    Bvec(int n_, bool verbose=false)
    {
	using sk_globals::mu_min;
	using sk_globals::mu_max;
	using sk_globals::n_min;
	
	this->n = n_;
	this->bias_coeffs = sk_globals::get_bias_coeffs();
	assert(n >= n_min);

	double eps = 5.0e-11;
	this->min_s1 = n * mu_min;
        this->min_s1 = int((1-eps)*min_s1 + 1);  // round up
        this->max_s1 = n * mu_max;
        this->max_s1 = int((1+eps)*max_s1);      // round down

	if (verbose) {
	    cout << "min_s1 = " << min_s1 << endl;
	    cout << "max_s1 = " << max_s1 << endl;
	}
	
	assert(min_s1 > 0);
	assert(min_s1 <= max_s1);
	assert(max_s1 <= 98*n);
	
	this->bvec = vector<T> (98*n+1, 0);

	// FIXME in hindsight, interpolation could be done more efficiently in two
	// steps. First, take the 2-d shape-(128,4) coefficient array, and "evaluate y"
	// to get a 1-d length-128 array. Then, "resample x" by interpolation, to get
	// a 1-d length-(98*n+1) array.

	for (int s1 = min_s1; s1 <= max_s1; s1++) {
	    double mu = s1/double(n);
	    double x = log(mu);
	    double y = 1.0/n;
	    bvec[s1] = _interpolate(x, y);

	    if (verbose)
		cout << "   bvec[" << s1 << "] = " << bvec[s1] << endl;
	}
    }
};



// FIXME move to gputils!
inline int quantize(double x)
{
    x = std::abs(x);
    int i = int(x + 0.5);
    return std::min(i, 7);
}


int main(int argc, char **argv)
{
    std::mt19937 &rng = gputils::default_rng;
    bool verbose = false;
    
    if (argc != 3) {
	cerr << "Usage: test-sk-bias <rms> <n>" << endl;
	return 2;
    }

    double rms = gputils::from_str<double> (argv[1]);
    int n = gputils::from_str<int> (argv[2]);
    cout << "Running Monte Carlos with rms=" << rms << ", n=" << n << endl;

    std::normal_distribution<double> dist(0, rms);
    Bvec<double> bvec(n, verbose);

    long nmc = 0;
    long ndenom = 0;
    double sum_sk = 0.0;
    double sum_sk2 = 0.0;
    int nbatch = (1<<21)/n + 1;

    // Run forever!
    for (int iouter = 1; 1; iouter++) {
	for (int b = 0; b < nbatch; b++) {
	    int s1 = 0;
	    int s2 = 0;

	    for (int i = 0; i < n; i++) {
		int ex = quantize(dist(rng));
		int ey = quantize(dist(rng));
		int e2 = ex*ex + ey*ey;
		s1 += e2;
		s2 += e2*e2;
	    }

	    ndenom++;
	    
	    if ((s1 < bvec.min_s1) || (s1 > bvec.max_s1))
		continue;

	    double ds1 = double(s1);
	    double sk = n*s2/(ds1*ds1) - 1;
	    sk *= (n+1) / double(n-1);
	    sk -= bvec.bvec[s1];

	    nmc++;
	    sum_sk += sk;
	    sum_sk2 += sk*sk;
	}

	if (!is_perfect_square(iouter) || (nmc < 2))
	    continue;

	double mean = sum_sk / nmc;
	double rms = sqrt((sum_sk2 - nmc*mean*mean) / double(nmc-1) / double(nmc));
	double pvalid = nmc / double(ndenom);

	cout << "nmc=" << nmc << ", pvalid=" << pvalid
	     << ", mean=" << mean << ", rms=" << rms << endl;
    }
}
