/* ranecu
 * based on the fortran code in the paper by F. James 1988
 *
 */

#include "seed.hpp"

namespace ranecu {

enum {
	num_randoms_per_call = 1
};

struct RNGState
{
	int z1;
	int z2;
};

struct DevParameters
{
	unsigned int *seeds;
};

inline void
initialize_rng(DevParameters *params)
{
	params->seeds = allocate_seeded_device_memory(sizeof(*params->seeds) * NUM_THREADS * 2);
	
}

inline void
destroy_rng(DevParameters *params)
{
	// free seeds
	CUDA_CHECK_ERROR(cudaFree(params->seeds));
	params->seeds = 0;
}

/**
 */
__device__ unsigned int
advance_ranecu(RNGState *state)
{
    unsigned int k;
    k = state->z1 / 53668;
    state->z1 = 40014 * (state->z1 - k * 53668) - k * 12211;
    if (state->z1 < 0) state->z1 = state->z1 + 2147483563;
	k = state->z2 / 52774;
    state->z2 = 40692 * (state->z2 - k * 52774) - k * 3791;
    if (state->z2 < 0) state->z2 = state->z2 + 2147483399;
    return state->z1 - state->z2;
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->z1 = params->seeds[idx * 2 + 0];
	state->z2 = params->seeds[idx * 2 + 1];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->seeds[idx * 2 + 0] = state->z1;
	params->seeds[idx * 2 + 1] = state->z2;
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = 0; i < num_randoms; i++) {
		out[i * stride] = advance_ranecu(state);
	}
}


} // namespace ranecu
