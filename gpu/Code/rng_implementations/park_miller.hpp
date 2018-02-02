#include "seed.hpp"

namespace park_miller {

enum {
	num_randoms_per_call = 1,
};

struct RNGState
{
	unsigned int seed;
};

struct DevParameters
{
	unsigned int *seeds;
};

#warning seeding broken, random seeding really bad here

inline void
initialize_rng(DevParameters *params)
{
	size_t s = sizeof(*params->seeds) * NUM_THREADS;
	CUDA_CHECK_ERROR(cudaMalloc(&params->seeds, s));

	unsigned int *tmp;
	cudaMallocHost(&tmp, s);
	seed_memory(tmp, s);
	cudaMemcpy(params->seeds, tmp, s, cudaMemcpyHostToDevice);
	cudaFreeHost(tmp);
}

inline void
destroy_rng(DevParameters *params)
{
	CUDA_CHECK_ERROR(cudaFree(params->seeds));
	params->seeds = 0;
}

__device__ unsigned int
advance_park_miller(RNGState *state)
{
	const double a = 16807.0;
	const double m = 2147483647.0;
	const double r_m = 1.0 / m;

	double tmp = state->seed * a;
	return state->seed = (unsigned int)(tmp - m * floor(tmp * r_m));
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->seed = params->seeds[idx];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->seeds[idx] = state->seed;
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	for(int i = 0; i < num_randoms; i++) {
		out[i * stride] = advance_park_miller(state);
	}
}

} // namespace park_miller
