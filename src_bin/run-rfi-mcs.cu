#include "../include/n2k.hpp"
#include "../include/n2k/interpolation.hpp"

#include <gputils/string_utils.hpp>
#include <iostream>

using namespace std;
using namespace n2k;


// FIXME move somewhere more general?
static bool is_perfect_square(int n)
{
    if (n < 1)
	return false;
    int m = int(sqrt(n) + 0.5);
    return n == m*m;
}

// FIXME move somewhere more general?
inline int quantize(double x)
{
    x = std::abs(x);
    int i = int(x + 0.5);
    return std::min(i, 7);
}


int main(int argc, char **argv)
{
    std::mt19937 &rng = gputils::default_rng;
    
    if (argc != 3) {
	cerr << "Usage: test-sk-bias <rms> <n>" << endl;
	return 2;
    }

    double rms = gputils::from_str<double> (argv[1]);
    long n = gputils::from_str<long> (argv[2]);
    cout << "Running Monte Carlos with rms=" << rms << ", n=" << n << endl;
    
    long min_s1 = n * sk_globals::mu_min;
    long max_s1 = n * sk_globals::mu_max;
    
    double eps = 5.0e-11;
    min_s1 = long((1-eps)*min_s1 + 1);  // round up
    max_s1 = long((1+eps)*max_s1);      // round down

    // Precompute biases for all possible values of S1.
    // (This is faster than doing interpolation in the inner loop.)

    vector<double> bvec(98*n+1, 0);
    for (long s1 = min_s1; s1 <= max_s1; s1++) {
	double x = log(double(s1) / double(n));
	double y = 1.0 / double(n);
	bvec[s1] = interpolate_bias_cpu(x,y);
    }

    long nmc = 0;
    double sum_sk = 0.0;
    double sum_sk2 = 0.0;
    std::normal_distribution<double> dist(0, rms);

    // Run forever!
    for (long nouter = 1; 1; nouter++) {
	long s1 = 0;
	long s2 = 0;

	for (long i = 0; i < n; i++) {
	    long ex = quantize(dist(rng));
	    long ey = quantize(dist(rng));
	    long e2 = ex*ex + ey*ey;
	    s1 += e2;
	    s2 += e2*e2;
	}
	    
	if ((s1 < min_s1) || (s1 > max_s1))
	    continue;

	double ds1 = double(s1);
	double sk = n*s2/(ds1*ds1) - 1;
	sk *= (n+1) / double(n-1);
	sk -= bvec[s1];
	
	nmc++;
	sum_sk += sk;
	sum_sk2 += sk*sk;

	if (!is_perfect_square(nmc) || (nmc < 2))
	    continue;

	double mean = sum_sk / nmc;
	double rms = sqrt((sum_sk2 - nmc*mean*mean) / double(nmc-1) / double(nmc));
	double pvalid = nmc / double(nouter);

	cout << "nmc=" << nmc << ", pvalid=" << pvalid
	     << ", mean=" << mean << ", rms=" << rms << endl;
    }
}
