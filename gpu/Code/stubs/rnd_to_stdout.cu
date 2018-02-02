#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <signal.h>
#include "util.hpp"
#include <unistd.h>

using namespace std;

unsigned int *random_numbers;
unsigned int *random_numbers_dev;

#define NUM_THREADS (1 << 14)
#define BLOCKSIZE 256
#define NUM_RANDOM_NUMBERS_DEV 2048

#include RANDOM_NUMBER_GENERATOR

enum { NUM_RANDOMS = NUM_THREADS * NUM_RANDOM_NUMBERS_DEV };

void
initialize_cuda()
{
	choose_device();

	CUDA_CHECK_ERROR(cudaMallocHost(&random_numbers,
				sizeof(*random_numbers) * NUM_RANDOMS));

	CUDA_CHECK_ERROR(cudaMalloc(&random_numbers_dev,
				sizeof(*random_numbers_dev) * NUM_RANDOMS));
}

__global__ void
kernel_generate_randoms(const RNG::DevParameters params, unsigned int *random_numbers)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	RNG::RNGState rng_state;
	RNG::initialize(&params, &rng_state);

	RNG::generate_random_numbers(&rng_state, random_numbers + idx, NUM_THREADS,
			NUM_RANDOM_NUMBERS_DEV);

	RNG::finalize(&params, &rng_state);
}

void
handle_sig_pipe(int sig)
{
	exit(EXIT_SUCCESS);
}

int
main(int argc, char **argv)
{
	if(isatty(1)) {
		cerr << "i won't write to a tty" << endl;
		exit(EXIT_FAILURE);
	}

	signal(SIGPIPE, handle_sig_pipe);

	initialize_cuda();

	RNG::DevParameters rng_parameters;
	RNG::initialize_rng(&rng_parameters);

	dim3 block(BLOCKSIZE, 1, 1);
	dim3 grid(NUM_THREADS / BLOCKSIZE, 1, 1);

	for(;;) {
		kernel_generate_randoms<<< grid, block >>> (rng_parameters, random_numbers_dev);
		CUDA_CHECK_ERROR(cudaGetLastError());
		CUDA_CHECK_ERROR(cudaMemcpy(random_numbers, random_numbers_dev,
					sizeof(*random_numbers_dev) * NUM_RANDOMS,
					cudaMemcpyDeviceToHost));

		unsigned int *ptr = random_numbers;
		int s = 0, cnt = sizeof(*ptr) * NUM_RANDOMS;
		while((s = write(1, ptr, cnt)) < cnt) {
			if(s < 0) {
				perror("error writing");
				exit(EXIT_FAILURE);
			}
			ptr += s;
			cnt -= s;
		}
	}


	return 0;
}
