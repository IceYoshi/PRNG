/*
 * Combined LCG and Tausworthe Generator from GPU Gems 3 p. 814
 * period is 2^121
 */
#include "seed.hpp"

namespace combinedLCGTaus {

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


__device__ unsigned int 
tausStep(unsigned int &z, unsigned int S1, unsigned int S2, unsigned int S3, unsigned int M)
{
	unsigned int b = (((z << S1) ^ z) >> S2);
	return z = (((z & M) << S3) ^ b);
}

__device__ unsigned int 
lcgStep(unsigned int &z, unsigned int A, unsigned int C) 
{
	return z = (A * z + C);
}

__device__ unsigned int
hybridTaus (RNGState *state)
{
	return tausStep(state->z1, 13, 19, 12, 4294967294UL) ^
		tausStep(state->z2, 2, 25, 4, 4294967288UL) ^
		tausStep(state->z3, 3, 11, 17, 4294967280UL) ^
		lcgStep(state->z4, 1664525, 1013904223UL);

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
		out[i * stride] = hybridTaus(state);
	}
}


} // namespace combinedLCGTaus
