#ifndef _N2K_DEVICE_INLINES_HPP
#define _N2K_DEVICE_INLINES_HPP

namespace n2k {
#if 0
}  // editor auto-indent
#endif


static constexpr uint FULL_MASK = 0xffffffffU;


// The nvidia LOP3 instruction.
// Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3
template<int immLut>
static __device__ int lop3(int a, int b, int c)
{
    int d;
    
    asm("lop3.b32 %0, %1, %2, %3, %4;" :
	"=r"(d) : "r"(a), "r"(b), "r"(c), "n"(immLut)
	);
    
    return d;
}


// blend(): equivalent to (c & a) | (~c & b), but using a single LOP3 instruction.    
static __device__ int blend(int a, int b, int c)
{
    constexpr int A = 0xf0;
    constexpr int B = 0xcc;
    constexpr int C = 0xaa;
    constexpr int N = (C & A) | (~C & B);
    
    return lop3<N> (a, b, c);
}


// Used in _transpose_bit_with_lane().
template<uint B> struct _bit_selector;
template<> struct _bit_selector<1> { static constexpr uint value = 0xaaaaaaaaU; };  // 0xa = (1010)_2
template<> struct _bit_selector<2> { static constexpr uint value = 0xccccccccU; };  // 0xc = (1100)_2
template<> struct _bit_selector<4> { static constexpr uint value = 0xf0f0f0f0U; };
template<> struct _bit_selector<8> { static constexpr uint value = 0xff00ff00U; };
template<> struct _bit_selector<16> { static constexpr uint value = 0xffff0000U; };


// Called by transpose_bit_with_lane()
// If upper: returns ( y[1], x[1] )
// If lower: returns ( x[0], y[0] )

template<uint B> __device__ uint _transpose_bit_with_lane(uint x, uint y, bool upper)
{
    static_assert((B==1) || (B==2) || (B==4) || (B==8) || (B==16));
    
    constexpr uint bs = _bit_selector<B>::value;
    uint b = upper ? bs : ~bs;
    uint z = upper ? (y >> B) : (y << B);
    return blend(x, z, b);

    // FIXME if B >= 8, do __byte_perm().
}


// Usage: transpose_bit_with_lane<16>(x,8) exchanges SIMD bit b_4
// (representing upper/lower halves of a 32-bit register) with thread bit b_3.

template<uint B>
__device__ uint transpose_bit_with_lane(uint x, uint lane)
{
    static_assert((B==1) || (B==2) || (B==4) || (B==8) || (B==16));
    
    bool upper = threadIdx.x & lane;
    uint y = __shfl_sync(FULL_MASK, x, threadIdx.x ^ lane);
    return _transpose_bit_with_lane<B> (x, y, upper);
}


}  // namespace n2k

#endif // _N2K_DEVICE_INLINES_HPP
