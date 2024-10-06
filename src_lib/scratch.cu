

template<bool Debug>
__global__ void __launch_bounds__(128,4)
correlate_pl_kernel_S128(int4 *V_out, const int2 *pl_mask, uint N128)
{
    if constexpr (Debug) {
	assert(blockDim.x == 128);
	assert(blockDim.y == 1);
	assert(blockDim.z == 1);
	assert(gridDim.x == 1);
    }
    
    constexpr int S = 16;    
    const uint tout = blockIdx.z * blockDim.z + threadIdx.z;
    const uint f = blockIdx.y * blockDim.y + threadIdx.y;
    const uint F = gridDim.y;
    
    const int w = threadIdx.x >> 5;
    const int wx = (w & 1);
    const int wy = (w >> 1);

    __shared__ int shmem[16*32];  // 2 KB
    
    // int2 pl_mask[T/64][F][S];
    // We assign a 32-by-128 submatrix to each warp.
    const uint t128_stride = (2*F) * S;       // 64-bit (int2) stride
    pl_mask += ulong(tout) * ulong(N128) * ulong(t128_stride);
    pl_mask += (f * S) + (w * 32) + pl_lane_offset(F*S);

    int pl_in[2][2][1];

    while (N128 > 0) {
	// Read 32-by-128 submatrix.
	read_pl_16_128(pl[0], pl_mask);
	read_pl_16_128(pl[1], pl_mask + 16);
	
	shmem[ + laneId] = pl[0][0][0];
	shmem[ + laneId] = pl[0][1][0];
	shmem[ + laneId] = pl[1][0][0];
	shmem[ + laneId] = pl[1][0][0];

	__syncthreads();

    }
}
