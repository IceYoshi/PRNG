#ifndef ROUNDS
# define ROUNDS 32
#endif

namespace tea {

enum {
	num_randoms_per_call = 2
};

struct RNGState
{
	unsigned int seed;
};

struct DevParameters
{
	unsigned int *seeds;
};

inline void
initialize_rng(DevParameters *params)
{
	// allocate seeds
	CUDA_CHECK_ERROR(cudaMalloc(&params->seeds, sizeof(*params->seeds) * NUM_THREADS));
	CUDA_CHECK_ERROR(cudaMemset(params->seeds, 0, sizeof(*params->seeds) * NUM_THREADS));
}

inline void
destroy_rng(DevParameters *params)
{
	// free seeds
	CUDA_CHECK_ERROR(cudaFree(params->seeds));
	params->seeds = 0;
}

__device__ void
encrypt_tea(unsigned int *arg)
{
	const unsigned int key[] = {
		0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e
	};
	unsigned int v0 = arg[0], v1 = arg[1];
	unsigned int sum = 0;
	unsigned int delta = 0x9e3779b9;

	#pragma unroll
	for(int i = 0; i < ROUNDS; i++) {
		sum += delta;
		v0 += ((v1 << 4) + key[0]) ^ (v1 + sum) ^ ((v1 >> 5) + key[1]);
		v1 += ((v0 << 4) + key[2]) ^ (v0 + sum) ^ ((v0 >> 5) + key[3]);
	}
	arg[0] = v0;
	arg[1] = v1;
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
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	while(num_randoms > 0) {
		unsigned int arg[2] = {
			idx, state->seed++
		};

		encrypt_tea(arg);

		#pragma unroll
		for(int i = 0; i < tea::num_randoms_per_call && i < num_randoms; i++) {
			*out = arg[i];
			out += stride;
			num_randoms--;
		}
	}
}


} // namespace tea
