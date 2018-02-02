#include "seed.hpp"

namespace tt800 {

enum {
	num_randoms_per_call = 1, // ?? maybe change
	N = 25,
	M = 7,
	A = 0x8ebfd028
};

struct RNGState
{
	unsigned int *seeds;
	unsigned int idx;
};

struct DevParameters
{
	unsigned int *seeds;
	unsigned int *indices;
};

inline void
initialize_rng(DevParameters *params)
{
	size_t s = sizeof(*params->seeds) * NUM_THREADS * N;
	CUDA_CHECK_ERROR(cudaMalloc(&params->seeds, s));
	CUDA_CHECK_ERROR(cudaMalloc(&params->indices, sizeof(*params->indices) * NUM_THREADS));
	cudaMemset(params->indices, 0xff, sizeof(*params->indices) * NUM_THREADS);
	

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
advance_tt800(RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(state->idx >= N) {
		#pragma unroll
		for(int k = 0; k < N - M; ++k) {
			int idx_k = k * NUM_THREADS;
			state->seeds[idx_k] = state->seeds[idx_k + M * NUM_THREADS] ^ (state->seeds[idx_k] >> 1) ^ ((state->seeds[idx_k] & 1) * A);
		}
		#pragma unroll
		for(int k = N - M; k < N; ++k) {
			int idx_k = k * NUM_THREADS;
			state->seeds[idx_k] = state->seeds[idx_k + (M - N) * NUM_THREADS] ^ (state->seeds[idx_k] >> 1) ^ ((state->seeds[idx_k] & 1) * A);
		}
		state->idx = 0;
	}

	unsigned int e = state->seeds[(state->idx++) * NUM_THREADS];
	e ^= (e << 7) & 0x2b5b2500;
	e ^= (e << 15) & 0xdb8b0000;
	e ^= (e >> 16);

	return e;
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->seeds = params->seeds + idx;
	state->idx = params->indices[idx];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->indices[idx] = state->idx;
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	for(int i = 0; i < num_randoms; i++) {
		out[i * stride] = advance_tt800(state);
	}
}

} // namespace tt800
