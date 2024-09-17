#ifndef _N2K_INTERNALS_HPP
#define _N2K_INTERNALS_HPP

#include <complex>
#include <gputils/Array.hpp>


namespace n2k {
#if 0
}  // editor auto-indent
#endif


extern gputils::Array<std::complex<int>> make_random_unpacked_e_array(int T, int F, int S);       // returns shape (T,F,S,2)
extern gputils::Array<std::complex<int>> unpack_e_array(const gputils::Array<uint8_t> &E_in);     // shape-preserving
extern gputils::Array<uint8_t> pack_e_array(const gputils::Array<std::complex<int>> &E_in);       // shape-preserving

extern gputils::Array<uint> make_random_s012_array(int T, int F, int S);   // returns shape (T,F,3,S)
extern gputils::Array<uint8_t> make_random_bad_feed_mask(int S);           // returns shape (S,)


extern void _check_array(int ndim, const ssize_t *shape, const ssize_t *strides, ssize_t size, int aflags,
			 const char *func_name, const char *arr_name, int expected_ndim, bool contiguous);

template<typename T>
inline void check_array(const gputils::Array<T> &arr, const char *func_name, const char *arr_name, int expected_ndim, bool contiguous)
{
    _check_array(arr.ndim, arr.shape, arr.strides, arr.size, arr.aflags, func_name, arr_name, expected_ndim, contiguous);
}


} // namespace n2k

#endif // _N2K_INTERNALS_HPP