
// -------------------------------------------------------------------------------------------------


// The shared memory pointer 'sp' has been arranged on each lane so that
//   (sp % 32) == (laneId & ~6)
//
// Returns sp[0] + sp[1]*y + sp[2]*y^2 + sp[3]*y^3.

template<bool Debug>
__device__ inline float b_inner(const float *sp, float y)
{
    int s = threadIdx.x & 6;
    
    float w0 = bank_conflict_free_load<Debug> (sp + s);
    float w1 = bank_conflict_free_load<Debug> (sp + ((s+2) & 6));
    float w2 = bank_conflict_free_load<Debug> (sp + ((s+4) & 6));
    float w3 = bank_conflict_free_load<Debug> (sp + ((s+6) & 6));

    roll4(threadIdx.x, w0, w1, w2, w3);
    return w0 + (w1*y) + (w2*y2) + (w3*y3);
    
}


__device__ inline float load_bias_coeffs(const float *bias_coeffs, int i, float y, float &c0, float &c1, float &c2, float &c3)
{
    bias_coeffs += (threadIdx.x & 1);

    int s = threadIdx.x & 
}

					 

// -------------------------------------------------------------------------------------------------


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

					  
__device__ inline float interp_sigma(const float *sigma_coeffs, int i, float dt)
{
    int ihi = ix & ~3;
    int s = 8*ihi + laneId;
    
    float c0 = sigma_coeffs[s];
    float c1 = sigma_coeffs[s+8];
    float c2 = sigma_coeffs[s+16];
    float c3 = sigma_coeffs[s+24];
}
