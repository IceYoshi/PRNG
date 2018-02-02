#include "seed.hpp"

/* name collides with stdlib */
namespace drand48gpu {

enum {
	num_randoms_per_call = 1,
	A0 = 0xE66D,
	A1 = 0xDEEC,
	A2 = 0x0005,
	C  = 0x000B
};

struct RNGState
{
	unsigned int seed[2];
};

struct DevParameters
{
	unsigned int *seeds;
};

inline void
initialize_rng(DevParameters *params)
{
	params->seeds = allocate_seeded_device_memory(
			sizeof(*params->seeds) * NUM_THREADS * 2);
}

inline void
destroy_rng(DevParameters *params)
{
	CUDA_CHECK_ERROR(cudaFree(params->seeds));
	params->seeds = 0;
}

__device__ unsigned int
advance_drand48(RNGState *state)
{
	int x0, x1, x2;

	x0 = state->seed[0] & 0xffff;
	x1 = state->seed[0] >> 16;
	x2 = state->seed[1] & 0xffff;

	int x;
   
	x = A0 * x0 + C;
	x0 = x & 0xffff;
	x >>= 16;

	x += A0 * x1 + A1 * x0;
	x1 = x & 0xffff;
	x >>= 16;

	x += A0 * x2 + A1 * x1 + A2 * x0;
	x2 = x & 0xffff;

	state->seed[0] = x0 | (x1 << 16);
	state->seed[1] = x2;

	return x;
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->seed[0] = params->seeds[idx];
	state->seed[1] = params->seeds[idx + NUM_THREADS];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->seeds[idx]               = state->seed[0];
	params->seeds[idx + NUM_THREADS] = state->seed[1];
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	while(num_randoms > 0) {
		*out = advance_drand48(state);
		out += stride;
		num_randoms--;
	}
}

} // namespace drand48gpu
