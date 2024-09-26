template<bool Debug>
__device__ inline float bank_conflict_free_load(const float *p)
{
    if constexpr (Debug) {
	int bank = ulong(p) & 31;
	uint bit = 1U << bank;
	uint bits = __reduce_or_sync(FULL_MASK, bit);
	assert(bits == FULL_MASK);
    }

    return *p;
}

__device__ inline float swap_if(bool flag, float &x, float &y)
{
    float t = x;
    x = flag ? y : x;
    y = flag ? t : y;
}

// xout[j] = xin[(j+i) % 4)]
__device__ inline float roll_forward(int i, float &x0, float &x1, float &x2, float &x3)
{
    bool flag2 = ((i & 2) != 0);
    swap_if(flag2, x0, x2);
    swap_if(flag2, x1, x3);

    bool flag1 = ((i & 1) != 0);
    float t = x0;
    
    x0 = flag1 ? x1 : x0;
    x1 = flag1 ? x2 : x1;
    x2 = flag1 ? x3 : x2;
    x3 = flag1 ? t : x3;
}

// xout[j] = xin[(j-i) % 4]
__device__ inline float roll_backward(int i, float &x0, float &x1, float &x2, float &x3)
{
    bool flag2 = ((i & 2) != 0);
    swap_if(flag2, x0, x2);
    swap_if(flag2, x1, x3);

    bool flag1 = ((i & 1) != 0);
    float t = x3;

    x3 = flag1 ? x2 : x3;
    x2 = flag1 ? x1 : x2;
    x1 = flag1 ? x0 : x1;
    x0 = flag1 ? t : x0;
}


// 'sigma_coeffs' points to an array of shape (N,8).
// The length-8 axis is a spectator, i.e. all 8 values are equal.
//
// This function is equivalent to:
//   c0 = sigma_coeffs[8*i];
//   c1 = sigma_coeffs[8*i+8];
//   c2 = sigma_coeffs[8*i+16];
//   c3 = sigma_coeffs[8*i+24];
//
// but is guaranteed bank conflict free.

template<bool Debug = false>
__device__ inline float load_sigma_coeffs(const float *sigma_coeffs, int i, float &c0, float &c1, float &c2, float &c3)
{
    int t = threadIdx.x;
    
    c0 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 24) & ~31) - t + 7);
    c1 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 16) & ~31) - t + 15);
    c2 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t + 8) & ~31) - t + 23);
    c3 = bank_conflict_free_load<Debug> (sigma_coeffs + ((8*i + t ) & ~31) - t + 31);

    roll_forward(i + (t >> 3), c0, c1, c2, c3);
}


// -------------------------------------------------------------------------------------------------


// Launch with 1 block and T threads.
//  out: shape (T,4)
//  sigma_coeffs: shape (N,8)
//  ix: shape (T,)

__global__ void sigma_coeffs_test_kernel(float *out, const float *sigma_coeffs, const int *ix)
{
    int t = threadIdx.x;
    load_sigma_coeffs<true> (sigma_coeffs, ix[t], out[4*t], out[4*t+1], out[4*t+2], out[4*t+3]);   // Debug=true
}


static void test_load_sigma_coeffs(int T, int N)
{
    cout << "test_load_sigma_coeffs(T=" << T << ", N=" << N << ")" << endl;
    
    Array<float> out({T,4}, af_uhost);
    Array<float> coeffs({N,8}, af_rhost);
    Array<int> ix({T}, af_rhost);

    for (int i = 0; i < N; i++) {
	float x = rand_uniform();
	for (int j = 0; j < 8; j++)
	    coeffs.at({i,j}) = x;
    }
    
    for (int t = 0; t < T; t++) {
	int i = rand_int(0, N-3);
	ix.at({t}) = i;
	
	for (int j = 0; j < 4; j++)
	    out.at({t,j}) = coeffs.at({i+j,0});
    }

    Array<float> out_gpu({T,4});
    Array<float> coeffs_gpu = coeffs.to_gpu();
    Array<int> ix_gpu = ix.to_gpu();

    sigma_coeffs_test_kernel<<<1,T>>> (out_gpu.data, coeffs_gpu.data, ix_gpu.data);
    CUDA_PEEK("sigma_coeffs_test_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(out, out_gpu, "cpu", "gpu", {"t","j"});
}


static void test_load_sigma_coeffs()
{
    for (int i = 0; i < 10; i++) {
	int T = 32 * rand_int(1,10);
	int N = rand_int(10, 100);
	test_load_sigma_coeffs(T, N);
    }
}


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
