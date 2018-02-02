/* 
 * lfsr113 from the paper "Tables of Maximally-Equidistributed Combined LFSR Generators", 
 * Piere L'Ecuyer, 1999
 * 
 * link: http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme2.ps
 */

#include "seed.hpp"

namespace lfsr113 {

enum {
	num_randoms_per_call = 1
};

struct RNGState
{
	unsigned int z1;
	unsigned int z2;
	unsigned int z3;
	unsigned int z4;
};

struct DevParameters
{
	unsigned int *seeds;
};

inline void
initialize_rng(DevParameters *params)
{
	size_t s = sizeof(*params->seeds) * NUM_THREADS * 4;
	CUDA_CHECK_ERROR(cudaMalloc(&params->seeds, s));

	unsigned int *tmp;
	cudaMallocHost(&tmp, s);
	seed_memory(tmp, s); // set all seed values randomly for each generator
	for (int i = 0; i < NUM_THREADS * 4; i++) {
		tmp[i] |= 128;   // makes sure each seed value is > 127 (this is required by the generator)
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

/**
 * The random number generation function for the lfsr113 generator (see top of file
 * for the source link). Uses a 128bit state to generate a new 32bit random number
 * in each call.
 */
__device__ unsigned int
advance_lfsr113(RNGState *state)
{
	unsigned int b;
	b = (((state->z1 << 6) ^ state->z1) >> 13);
	state->z1 = ((state->z1 & 4294967294UL) << 18) ^ b;
	b  = ((state->z2 << 2) ^ state->z2) >> 27; 
	state->z2 = ((state->z2 & 4294967288UL) << 2) ^ b;
	b  = ((state->z3 << 13) ^ state->z3) >> 21;
	state->z3 = ((state->z3 & 4294967280UL) << 7) ^ b;
	b  = ((state->z4 << 3) ^ state->z4) >> 12;
	state->z4 = ((state->z4 & 4294967168UL) << 13) ^ b;
	return (state->z1 ^ state->z2 ^ state->z3 ^ state->z4);
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->z1 = params->seeds[idx * 4 + 0];
	state->z2 = params->seeds[idx * 4 + 1];
	state->z3 = params->seeds[idx * 4 + 2];
	state->z4 = params->seeds[idx * 4 + 3];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->seeds[idx * 4 + 0] = state->z1;
	params->seeds[idx * 4 + 1] = state->z2;
	params->seeds[idx * 4 + 2] = state->z3;
	params->seeds[idx * 4 + 3] = state->z4;
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < num_randoms; i++) {
		out[i * stride] = advance_lfsr113(state);
	}
}


} // namespace lfsr113
