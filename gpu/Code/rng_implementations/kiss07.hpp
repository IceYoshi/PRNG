/* 
 * KISS rng from
 * http://groups.google.com/group/comp.lang.fortran/msg/6edb8ad6ec5421a5
 * period: 2^121
 */

#include "seed.hpp"

namespace kiss07 {

enum {
	num_randoms_per_call = 1
};

struct RNGState
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
	unsigned int c;
};

struct DevParameters
{
	unsigned int *seeds;
};

inline void
initialize_rng(DevParameters *params)
{
	size_t s = sizeof(*params->seeds) * NUM_THREADS * 5;
	CUDA_CHECK_ERROR(cudaMalloc(&params->seeds, s));

	unsigned int *tmp;
	cudaMallocHost(&tmp, s);
	seed_memory(tmp, s); // set all seed values randomly for each generator
	for (int i = 0; i < NUM_THREADS * 5; i+=5) {
		if (tmp[i+1] == 0) tmp[i+1] = 1;
		tmp[i+2] &=0x7FFFFFFF; // z
		tmp[i+3] &=0x7FFFFFFF; // w
		tmp[i+4] &=0x1; // w
	}
	cudaMemcpy(params->seeds, tmp, s, cudaMemcpyHostToDevice);
	cudaFreeHost(tmp);
}

inline void
destroy_rng(DevParameters *params)
{
	// free seeds
	CUDA_CHECK_ERROR(cudaFree(params->seeds));
	params->seeds = 0;
}

__device__ unsigned int
advance_kiss(RNGState *state)
{
	unsigned int t;
	state->x += 545925293;
	state->y ^= (state->y << 13); 
	state->y ^= (state->y >> 17); 
	state->y ^= (state->y << 5);
	t = state->z + state->w + state->c; 
	state->z = state->w; 
	state->c = (t >> 31); 
	state->w = t & 2147483647;
	return state->x + state->y + state->w;     
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->x = params->seeds[idx * 5 + 0];
	state->y = params->seeds[idx * 5 + 1];
	state->z = params->seeds[idx * 5 + 2];
	state->w = params->seeds[idx * 5 + 3];
	state->c = params->seeds[idx * 5 + 4];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->seeds[idx * 5 + 0] = state->x;
	params->seeds[idx * 5 + 1] = state->y;
	params->seeds[idx * 5 + 2] = state->z;
	params->seeds[idx * 5 + 3] = state->w;
	params->seeds[idx * 5 + 4] = state->c;
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < num_randoms; i++) {
		out[i * stride] = advance_kiss(state);
	}
}


} // namespace kiss07
