#include "../include/n2k/internals.hpp"
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


// FIXME improve!
Array<uint> make_random_s012_array(int T, int F, int S)
{
    std::mt19937 &rng = gputils::default_rng;
    auto dist = std::uniform_int_distribution<uint>(0, 1000);

    Array<uint> ret({T,F,3,S}, af_rhost);
    for (long i = 0; i < ret.size; i++)
	ret.data[i] = dist(rng);

    return ret;
}


Array<uint8_t> make_random_bad_feed_mask(int S)
{
    assert(S > 0);
    assert(S < 32*1024);   // FIXME should be a global somewhere
    assert(S % 128 == 0);  // assumed by load_bad_feed_mask()
    
    std::mt19937 &rng = gputils::default_rng;
    auto dist = std::uniform_int_distribution<uint8_t>(0,1);

    Array<uint8_t> ret({S}, af_rhost);
    for (long i = 0; i < S; i++)
	ret.data[i] = dist(rng);

    return ret;
}


Array<complex<int>> make_random_unpacked_e_array(int T, int F, int S)
{
    std::mt19937 &rng = gputils::default_rng;
    auto dist = std::uniform_int_distribution<int> (-7, 7);

    Array<complex<int>> ret({T,F,S}, af_uhost);
    for (long i = 0; i < ret.size; i++)
	ret.data[i] = { dist(rng), dist(rng) };

    return ret;
}


Array<complex<int>> unpack_e_array(const Array<uint8_t> &E_in)
{
    assert(E_in.on_host());
    assert(E_in.is_fully_contiguous());

    Array<complex<int>> E_out(E_in.ndim, E_in.shape, af_uhost);

    for (long i = 0; i < E_in.size; i++) {
	uint8_t e = E_in.data[i] ^ 0x88;   // twos complement -> offset-encode
	int e_re = int(e & 0xf) - 8;
	int e_im = int((e >> 4) & 0xf) - 8;       
	E_out.data[i] = { e_re, e_im };
    }

    return E_out;
}


Array<uint8_t> pack_e_array(const Array<complex<int>> &E_in)
{
    assert(E_in.on_host());
    assert(E_in.is_fully_contiguous());

    Array<uint8_t> E_out(E_in.ndim, E_in.shape, af_uhost);

    for (long i = 0; i < E_in.size; i++) {
	uint8_t e_re = E_in.data[i].real() & 0xf;
	uint8_t e_im = E_in.data[i].imag() & 0xf;
	E_out.data[i] = (e_re) | (e_im << 4);
    }

    return E_out;
}


void _check_array(int ndim, const ssize_t *shape, const ssize_t *strides, ssize_t size, int aflags,
		  const char *func_name, const char *arr_name, int expected_ndim, bool contiguous)
{
    const char *err;

    if (size == 0)
	err = "is empty";
    else if (expected_ndim != ndim)
	err = "does not have correct dimension";
    else if (!gputils::af_on_gpu(aflags))
	err = "is not on GPU";
    else if (contiguous && (gputils::compute_ncontig(ndim,shape,strides) != ndim))
	err = "is not contiguous";
    else
	return;

    stringstream ss;
    ss << func_name << "(): '" << arr_name << "' array " << err;
    throw runtime_error(ss.str());
}


}  // namespace n2k
