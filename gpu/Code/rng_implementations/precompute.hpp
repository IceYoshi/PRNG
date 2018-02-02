namespace precompute {

enum {
	num_randoms_per_call = 1,
	NUM_PRECOMPUTED_RANDOMS_PER_THREAD = 4096,
};

struct RNGState
{
	int pos;
	const unsigned int *random_numbers;
};

struct DevParameters
{
	int *seed_positions;
	unsigned int *random_numbers;
};

inline void
initialize_rng(DevParameters *params)
{
	size_t srandoms = NUM_PRECOMPUTED_RANDOMS_PER_THREAD * NUM_THREADS
		* sizeof(*params->random_numbers);
	size_t sindices = NUM_THREADS * sizeof(*params->seed_positions);

	CUDA_CHECK_ERROR(cudaMalloc(&params->random_numbers, srandoms));
	CUDA_CHECK_ERROR(cudaMalloc(&params->seed_positions, sindices));
	CUDA_CHECK_ERROR(cudaMemset(params->seed_positions, 0, sindices));

	/* upload random numbers here */
}

inline void
destroy_rng(DevParameters *params)
{
	cudaFree(params->seed_positions);
	cudaFree(params->random_numbers);
}

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	state->pos = params->seed_positions[idx];
	state->random_numbers = params->random_numbers + idx;
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	params->seed_positions[idx] = state->pos;
}

__device__ void
generate_random_numbers(RNGState *state, unsigned int *out, int stride, int num_randoms)
{
	for(int i = 0; i < num_randoms; i++) {
		state->pos = (state->pos + 1) % NUM_PRECOMPUTED_RANDOMS_PER_THREAD;
		out[i * stride] = state->random_numbers[__umul24(state->pos, NUM_THREADS)];
	}
}


} // namespace precompute

