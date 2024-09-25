__device__ inline float swap_if(bool flag, float &x, float &y)
{
    float t = x;
    x = flag ? y : x;
    y = flag ? t : y;
}

__device__ inline float roll4(int i, float &x0, float &x1, float &x2, float &x3)
{
    bool flag2 = ((i & 2) != 0);
    swap_if(flag2, x0, x2);
    swap_if(flag2, x1, x3);

    bool flag1 = ((i & 1) != 0);
    xxx;
}

// The shared memory pointer 'sp' has been arranged on each lane so that
//   (sp % 32) == (laneId & ~3)
//
// Returns sp[0] + sp[1]*y + sp[2]*y^2 + sp[3]*y^3

__device__ inline float eval_y_poly(const float *sp, float y)
{
    float w0 = sp[(threadIdx.x) & 3];
    float w1 = sp[(threadIdx.x+1) & 3];
    float w2 = sp[(threadIdx.x+2) & 3];
    float w3 = sp[(threadIdx.x+3) & 3];

    roll4(threadIdx.x, w0, w1, w2, w3);
    return w0 + (w1*y) + (w2*y2) + (w3*y3);
}


// The shared memory pointer 'sp' as been arranged on each lane so that
//   (sp % 32) == (laneId & 0x10)
//
// Returns 2-d interpolation based on sp[0:16].

__device__ inline float eval_xy_spline(const float *sp, float dx, float y)
{
    float c0 = eval_y_poly(sp + (threadIdx.x) & 0xc);
    float c1 = eval_y_poly(sp + (threadIdx.x+4) & 0xc);
    float c2 = eval_y_poly(sp + (threadIdx.x+8) & 0xc);
    float c3 = eval_y_poly(sp + (threadIdx.x+12) & 0xc);

    roll4(threadIdx.x >> 2, c0, c1, c2, c3);
    xxx;  // spline
}


__device__ inline 
