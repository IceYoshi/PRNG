#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <errno.h>
#include "util.hpp"

using namespace std;

#ifndef RAW_PERFORMANCE_NUM_THREADS
# error RAW_PERFORMANCE_NUM_THREADS not defined, fix your config.mk
#endif

#ifndef RAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD
# error RAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD not defined, fix your config.mk
#endif

#ifndef RAW_PERFORMANCE_NUM_KERNEL_CALLS
# error RAW_PERFORMANCE_NUM_KERNEL_CALLS not defined, fix your config.mk
#endif

#define BLOCKSIZE 256

unsigned int *random_numbers_dev;
enum { NUM_THREADS = RAW_PERFORMANCE_NUM_THREADS };

#include RANDOM_NUMBER_GENERATOR

//enum { NUM_RANDOMS = NUM_THREADS * RNG::num_randoms_per_call };


void
initialize_cuda()
{
	choose_device();

	CUDA_CHECK_ERROR(cudaMalloc(&random_numbers_dev, sizeof(*random_numbers_dev) * NUM_THREADS));
}

__global__ void
kernel_empty()
{
}

__global__ void
kernel_generate_randoms(const RNG::DevParameters params, unsigned int *random_numbers)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	RNG::RNGState rng_state;
	RNG::initialize(&params, &rng_state);

	unsigned int rnds[RNG::num_randoms_per_call];
	unsigned int sum = 0;

	for(int i = 0; i < RAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD / RNG::num_randoms_per_call; i++) {
		RNG::generate_random_numbers(&rng_state, rnds, 1, RNG::num_randoms_per_call);

		// use the random numbers to prevent the compiler from trying to be smart
		#pragma unroll
		for(int j = 0; j < RNG::num_randoms_per_call; j++)
			sum ^= rnds[j];
	}

	RNG::finalize(&params, &rng_state);
	random_numbers[idx] = sum;
}

int
main(int argc, char **argv)
{
	struct timeval tv1, tv2;
	initialize_cuda();

	RNG::DevParameters rng_parameters;
	RNG::initialize_rng(&rng_parameters);

	dim3 block(BLOCKSIZE, 1, 1);
	dim3 grid(RAW_PERFORMANCE_NUM_THREADS / BLOCKSIZE, 1, 1);

	kernel_empty<<< grid, block >>>();
	CUDA_CHECK_ERROR(cudaGetLastError());
	kernel_generate_randoms<<< grid, block >>> (rng_parameters, random_numbers_dev);
	CUDA_CHECK_ERROR(cudaGetLastError());
	cudaThreadSynchronize();

	gettimeofday(&tv1, NULL);
	// multiple kernel calls to prevent hitting the maximum kernel launch time
	for(int i = 0; i < RAW_PERFORMANCE_NUM_KERNEL_CALLS; i++) {
		kernel_generate_randoms<<< grid, block >>> (rng_parameters, random_numbers_dev);
		CUDA_CHECK_ERROR(cudaGetLastError());
	}
	cudaThreadSynchronize();
	gettimeofday(&tv2, NULL);

	cout << ((tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec)
		<< endl;

	return 0;
}
