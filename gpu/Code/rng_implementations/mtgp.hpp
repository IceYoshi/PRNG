/*
 * from M.Saito's original Implementation of the MTGP algorithm.
 * implementation
 */

#include <inttypes.h>

#ifndef UINT32_C
# define UINT32_C(a) (a)
#endif

namespace mtgp {

/**
 * Sample Program for CUDA 2.3
 * written by M.Saito (saito@math.sci.hiroshima-u.ac.jp)
 *
 * This sample uses texture reference.
 * The generation speed of PRNG using texture is faster than using
 * constant tabel on Geforce GTX 260.
 *
 * MTGP32-11213
 * This program generates 32-bit unsigned integers.
 * The period of generated integers is 2<sup>11213</sup>-1.
 * This also generates single precision floating point numbers.
 *
 * Copyright (C) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and
 * Hiroshima University. All rights reserved.
 */

#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGPDC_PARAM_TABLE mtgp32dc_params_fast_11213


struct mtgp32_status_fast_t {
    int idx;			/**< index */
    int size;			/**< minimum needed size */
    int large_size;		/**< real size of array */
    int large_mask;		/**< bit mask to update \b idx */
    unsigned int array[];		/**< internal state array */
};

struct mtgp32_params_fast_t {
    int mexp;			/**< Mersenne exponent. This is redundant. */
    int pos;			/**< pick up position. */
    int sh1;			/**< shift value 1. 0 < sh1 < 32. */
    int sh2;			/**< shift value 2. 0 < sh2 < 32. */
    unsigned int tbl[16];		/**< a small matrix. */
    unsigned int tmp_tbl[16];	/**< a small matrix for tempering. */
    unsigned int flt_tmp_tbl[16];	/**< a small matrix for tempering and converting to float. */
    unsigned int mask;		/**< This is a mask for state space */
    unsigned char poly_sha1[21]; /**< SHA1 digest */
};

struct mtgp32_fast_t {
    mtgp32_params_fast_t params; /**< parameters */
    mtgp32_status_fast_t *status; /**< internal state */
};

#include "mtgp_table.hpp"

enum {
	MEXP                 = 11213,
	N                    = MTGPDC_N,
	THREAD_NUM           = MTGPDC_FLOOR_2P,
	LARGE_SIZE           = THREAD_NUM * 3,
	BLOCK_NUM_MAX        = 200,
	TBL_SIZE             = 16,
	num_randoms_per_call = 3,
};

/* MTGPDC_FLOOR_2P == THREAD_NUM!!! */
#if BLOCKSIZE != MTGPDC_FLOOR_2P
# error BLOCKSIZE must be the same as THREAD_NUM
#endif

struct mtgp32_kernel_status_t {
	uint32_t status[N];
};

struct RNGState
{
	int pos;
};

struct DevParameters
{
	unsigned int *status;
	mtgp32_kernel_status_t *d_status;
};
/* {{{ stuff */
/**
 * kernel I/O
 * This structure must be initialized before first use.
 */
uint32_t *d_texture[3];

texture<uint32_t, 1, cudaReadModeElementType> tex_param_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_temper_ref;
texture<uint32_t, 1, cudaReadModeElementType> tex_single_ref;
/*
 * Generator Parameters.
 */
__constant__ uint32_t pos_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh1_tbl[BLOCK_NUM_MAX];
__constant__ uint32_t sh2_tbl[BLOCK_NUM_MAX];
/* high_mask and low_mask should be set by make_constant(), but
 * did not work.
 */
__constant__ uint32_t mask = 0xff800000;

/**
 * Shared memory
 * The generator's internal status vector.
 */
__shared__ uint32_t status[LARGE_SIZE];

/**
 * The function of the recursion formula calculation.
 *
 * @param[in] X1 the farthest part of state array.
 * @param[in] X2 the second farthest part of state array.
 * @param[in] Y a part of state array.
 * @param[in] bid block id.
 * @return output
 */
__device__ unsigned int
para_rec(unsigned int X1, unsigned int X2, unsigned int Y, int bid) {
	unsigned int X = (X1 & mask) ^ X2;
	unsigned int MAT;

	X ^= X << sh1_tbl[bid];
	Y = X ^ (Y >> sh2_tbl[bid]);
	MAT = tex1Dfetch(tex_param_ref, bid * 16 + (Y & 0x0f));
	return Y ^ MAT;
}

/**
 * The tempering function.
 *
 * @param[in] V the output value should be tempered.
 * @param[in] T the tempering helper value.
 * @param[in] bid block id.
 * @return the tempered value.
 */
__device__ unsigned int
temper(unsigned int V, unsigned int T, int bid)
{
	unsigned int MAT;

	T ^= T >> 16;
	T ^= T >> 8;
	MAT = tex1Dfetch(tex_temper_ref, bid * 16 + (T & 0x0f));
	return V ^ MAT;
}

/**
 * Read the internal state vector from kernel I/O data, and
 * put them into shared memory.
 *
 * @param[out] status shared memory.
 * @param[in] d_status kernel I/O data
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void
status_read(uint32_t status[LARGE_SIZE], const mtgp32_kernel_status_t *d_status, int bid, int tid)
{
	status[LARGE_SIZE - N + tid] = d_status[bid].status[tid];
	if (tid < N - THREAD_NUM) {
		status[LARGE_SIZE - N + THREAD_NUM + tid]
			= d_status[bid].status[THREAD_NUM + tid];
	}
	__syncthreads();
}

/**
 * Read the internal state vector from shared memory, and
 * write them into kernel I/O data.
 *
 * @param[out] d_status kernel I/O data
 * @param[in] status shared memory.
 * @param[in] bid block id
 * @param[in] tid thread id
 */
__device__ void
status_write(mtgp32_kernel_status_t *d_status, const uint32_t status[LARGE_SIZE], int bid, int tid)
{
	d_status[bid].status[tid] = status[LARGE_SIZE - N + tid];
	if (tid < N - THREAD_NUM) {
		d_status[bid].status[THREAD_NUM + tid]
			= status[4 * THREAD_NUM - N + tid];
	}
	__syncthreads();
}

#if 0
/**
 * kernel function.
 * This function generates 32-bit unsigned integers in d_data
 *
 * @params[in,out] d_status kernel I/O data
 * @params[out] d_data output
 * @params[in] size number of output data requested.
 */
__global__ void
mtgp32_uint32_kernel(mtgp32_kernel_status_t *d_status, uint32_t* d_data, int size)
{
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	int pos = pos_tbl[bid];
	uint32_t r;
	uint32_t o;

	// copy status data from global memory to shared memory.
	status_read(status, d_status, bid, tid);

	// main loop
	for (int i = 0; i < size; i += LARGE_SIZE) {
		r = para_rec(status[LARGE_SIZE - N + tid],
				status[LARGE_SIZE - N + tid + 1],
				status[LARGE_SIZE - N + tid + pos],
				bid);
		status[tid] = r;

		o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);

		d_data[size * bid + i + tid] = o;
		__syncthreads();
		r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
				bid);
		status[tid + THREAD_NUM] = r;
		o = temper(r,
				status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
				bid);
		d_data[size * bid + THREAD_NUM + i + tid] = o;
		__syncthreads();
		r = para_rec(status[2 * THREAD_NUM - N + tid],
				status[2 * THREAD_NUM - N + tid + 1],
				status[2 * THREAD_NUM - N + tid + pos],
				bid);
		status[tid + 2 * THREAD_NUM] = r;
		o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
		d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;
		__syncthreads();
	}
	// write back status for next call
	status_write(d_status, status, bid, tid);
}
#endif

__device__ void
generate_random_numbers(RNGState *rng_state, unsigned int *out, int stride, int num_randoms)
{
	int pos = rng_state->pos;
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	uint32_t r;
	uint32_t o;
	while(num_randoms > 0) {
		r = para_rec(status[LARGE_SIZE - N + tid],
				status[LARGE_SIZE - N + tid + 1],
				status[LARGE_SIZE - N + tid + pos],
				bid);
		status[tid] = r;

		o = temper(r, status[LARGE_SIZE - N + tid + pos - 1], bid);

		//d_data[size * bid + i + tid] = o;
		*out = o;
		out += stride;
		num_randoms--;

		__syncthreads();
		r = para_rec(status[(4 * THREAD_NUM - N + tid) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + 1) % LARGE_SIZE],
				status[(4 * THREAD_NUM - N + tid + pos) % LARGE_SIZE],
				bid);
		status[tid + THREAD_NUM] = r;
		o = temper(r,
				status[(4 * THREAD_NUM - N + tid + pos - 1) % LARGE_SIZE],
				bid);
		//
		//d_data[size * bid + THREAD_NUM + i + tid] = o;

		if(num_randoms > 0) {
			*out = o;
			out += stride;
			num_randoms--;
		}

		__syncthreads();
		r = para_rec(status[2 * THREAD_NUM - N + tid],
				status[2 * THREAD_NUM - N + tid + 1],
				status[2 * THREAD_NUM - N + tid + pos],
				bid);
		status[tid + 2 * THREAD_NUM] = r;
		o = temper(r, status[tid + pos - 1 + 2 * THREAD_NUM - N], bid);
		//d_data[size * bid + 2 * THREAD_NUM + i + tid] = o;

		if(num_randoms > 0) {
			*out = o;
			out += stride;
			num_randoms--;
		}
		__syncthreads();
	}
}

void
mtgp32_init_state(uint32_t array[], const mtgp32_params_fast_t *para, uint32_t seed)
{
    int i;
    int size = para->mexp / 32 + 1;
    uint32_t hidden_seed;
    uint32_t tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(uint32_t) * size);
    array[0] = seed;
    array[1] = hidden_seed;
    for (i = 1; i < size; i++) {
		array[i] ^= UINT32_C(1812433253) * (array[i - 1] ^ (array[i - 1] >> 30)) + i;
    }
}

void
make_kernel_data(mtgp32_kernel_status_t *d_status, mtgp32_params_fast_t params[], int block_num)
{
	mtgp32_kernel_status_t* h_status = new mtgp32_kernel_status_t[block_num];

	for (int i = 0; i < block_num; i++) {
		mtgp32_init_state(&(h_status[i].status[0]), &params[i], i + 1);
	}
	CUDA_CHECK_ERROR(cudaMemcpy(d_status, h_status, sizeof(mtgp32_kernel_status_t) * block_num, cudaMemcpyHostToDevice));

	delete[] h_status;
}

/**
 * This function sets constants in device memory.
 * @param params input, MTGP32 parameters.
 */
void
make_constant_param(const mtgp32_params_fast_t params[], int block_num)
{
	const int size1 = sizeof(uint32_t) * block_num;
	unsigned int *h_pos_tbl = new unsigned int[block_num];
	unsigned int *h_sh1_tbl = new unsigned int[block_num];
	unsigned int *h_sh2_tbl = new unsigned int[block_num];

	for (int i = 0; i < block_num; i++) {
		h_pos_tbl[i] = params[i].pos;
		h_sh1_tbl[i] = params[i].sh1;
		h_sh2_tbl[i] = params[i].sh2;
	}
	// copy from malloc area only
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(pos_tbl, h_pos_tbl, size1));
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(sh1_tbl, h_sh1_tbl, size1));
	CUDA_CHECK_ERROR(cudaMemcpyToSymbol(sh2_tbl, h_sh2_tbl, size1));

	delete[] h_pos_tbl;
	delete[] h_sh1_tbl;
	delete[] h_sh2_tbl;
}

/**
 * This function sets texture lookup table.
 * @param params input, MTGP32 parameters.
 * @param d_texture_tbl device memory used for texture bind
 * @param block_num block number used for kernel call
 */
void
make_texture(const mtgp32_params_fast_t params[], uint32_t *d_texture_tbl[3], int block_num)
{
	const int count = block_num * TBL_SIZE;
	const int size = sizeof(uint32_t) * count;
	unsigned int *h_texture_tbl[3];

	for(int i = 0; i < 3; i++)
		h_texture_tbl[i] = new unsigned int[count];

	for(int i = 0; i < block_num; i++) {
		for (int j = 0; j < TBL_SIZE; j++) {
			h_texture_tbl[0][i * TBL_SIZE + j] = params[i].tbl[j];
			h_texture_tbl[1][i * TBL_SIZE + j] = params[i].tmp_tbl[j];
			h_texture_tbl[2][i * TBL_SIZE + j] = params[i].flt_tmp_tbl[j];
		}
	}

	CUDA_CHECK_ERROR(cudaMemcpy(d_texture_tbl[0], h_texture_tbl[0], size, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(d_texture_tbl[1], h_texture_tbl[1], size, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(d_texture_tbl[2], h_texture_tbl[2], size, cudaMemcpyHostToDevice));
	tex_param_ref.filterMode = cudaFilterModePoint;
	tex_temper_ref.filterMode = cudaFilterModePoint;
	tex_single_ref.filterMode = cudaFilterModePoint;
	CUDA_CHECK_ERROR(cudaBindTexture(0, tex_param_ref, d_texture_tbl[0], size));
	CUDA_CHECK_ERROR(cudaBindTexture(0, tex_temper_ref, d_texture_tbl[1], size));
	CUDA_CHECK_ERROR(cudaBindTexture(0, tex_single_ref, d_texture_tbl[2], size));

	for(int i = 0; i < 3; i++)
		delete[] h_texture_tbl[i];
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
#if 0
void make_uint32_random(mtgp32_kernel_status_t* d_status,
		int num_data,
		int block_num) {
	uint32_t* d_data;
	unsigned int timer = 0;
	uint32_t* h_data;
	cudaError_t e;
	float gputime;

	printf("generating 32-bit unsigned random numbers.\n");
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(uint32_t) * num_data));
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	h_data = (uint32_t *) malloc(sizeof(uint32_t) * num_data);
	if (h_data == NULL) {
		printf("failure in allocating host memory for output data.\n");
		exit(1);
	}
	CUT_SAFE_CALL(cutStartTimer(timer));
	if (cudaGetLastError() != cudaSuccess) {
		printf("error has been occured before kernel call.\n");
		exit(1);
	}

	/* kernel call */
	mtgp32_uint32_kernel<<< block_num, THREAD_NUM>>>(
			d_status, d_data, num_data / block_num);
	cudaThreadSynchronize();

	e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
		exit(1);
	}
	CUT_SAFE_CALL(cutStopTimer(timer));
	CUDA_SAFE_CALL(
			cudaMemcpy(h_data,
				d_data,
				sizeof(uint32_t) * num_data,
				cudaMemcpyDeviceToHost));
	gputime = cutGetTimerValue(timer);
	print_uint32_array(h_data, num_data, block_num);
	printf("generated numbers: %d\n", num_data);
	printf("Processing time: %f (ms)\n", gputime);
	printf("Samples per second: %E \n", num_data / (gputime * 0.001));
	CUT_SAFE_CALL(cutDeleteTimer(timer));
	//free memories
	free(h_data);
	CUDA_SAFE_CALL(cudaFree(d_data));
}

/**
 * host function.
 * This function calls corresponding kernel function.
 *
 * @param d_status kernel I/O data.
 * @param num_data number of data to be generated.
 */
void make_single_random(mtgp32_kernel_status_t* d_status,
		int num_data,
		int block_num) {
	float* d_data;
	unsigned int timer = 0;
	float* h_data;
	cudaError_t e;
	float gputime;

	printf("generating single precision floating point random numbers.\n");
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, sizeof(float) * num_data));
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	h_data = (float *) malloc(sizeof(float) * num_data);
	if (h_data == NULL) {
		printf("failure in allocating host memory for output data.\n");
		exit(1);
	}
	CUT_SAFE_CALL(cutStartTimer(timer));
	if (cudaGetLastError() != cudaSuccess) {
		printf("error has been occured before kernel call.\n");
		exit(1);
	}

	/* kernel call */
	mtgp32_single_kernel<<< block_num, THREAD_NUM >>>(
			d_status, d_data, num_data / block_num);
	cudaThreadSynchronize();

	e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("failure in kernel call.\n%s\n", cudaGetErrorString(e));
		exit(1);
	}
	CUT_SAFE_CALL(cutStopTimer(timer));
	CUDA_SAFE_CALL(
			cudaMemcpy(h_data,
				d_data,
				sizeof(uint32_t) * num_data,
				cudaMemcpyDeviceToHost));
	gputime = cutGetTimerValue(timer);
	print_float_array(h_data, num_data, block_num);
	printf("generated numbers: %d\n", num_data);
	printf("Processing time: %f (ms)\n", gputime);
	printf("Samples per second: %E \n", num_data / (gputime * 0.001));
	CUT_SAFE_CALL(cutDeleteTimer(timer));
	//free memories
	free(h_data);
	CUDA_SAFE_CALL(cudaFree(d_data));
}

int main(int argc, char *argv[])
{
	// LARGE_SIZE is a multiple of 16
	int num_data = 10000000;
	int block_num;
	int num_unit;
	int r;
	mtgp32_kernel_status_t *d_status;
	uint32_t *d_texture[3];

	if (argc >= 2) {
		errno = 0;
		block_num = strtol(argv[1], NULL, 10);
		if (errno) {
			printf("%s number_of_block number_of_output\n", argv[0]);
			return 1;
		}
		if (block_num < 1 || block_num > BLOCK_NUM_MAX) {
			printf("%s block_num should be between 1 and %d\n",
					argv[0], BLOCK_NUM_MAX);
			return 1;
		}
		errno = 0;
		num_data = strtol(argv[2], NULL, 10);
		if (errno) {
			printf("%s number_of_block number_of_output\n", argv[0]);
			return 1;
		}
		argc -= 2;
		argv += 2;
	} else {
		CUT_DEVICE_INIT(argc, argv);
		printf("%s number_of_block number_of_output\n", argv[0]);
		block_num = get_suitable_block_num(sizeof(uint32_t),
				THREAD_NUM,
				LARGE_SIZE);
		if (block_num <= 0) {
			printf("can't calculate sutable number of blocks.\n");
			return 1;
		}
		printf("the suitable number of blocks for device 0 "
				"will be multiple of %d\n", block_num);
		return 1;
	}
	CUT_DEVICE_INIT(argc, argv);

	num_unit = LARGE_SIZE * block_num;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_status,
				sizeof(mtgp32_kernel_status_t) * block_num));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_texture[0],
				sizeof(uint32_t) * block_num * TBL_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_texture[1],
				sizeof(uint32_t) * block_num * TBL_SIZE));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_texture[2],
				sizeof(uint32_t) * block_num * TBL_SIZE));
	r = num_data % num_unit;
	if (r != 0) {
		num_data = num_data + num_unit - r;
	}

	make_constant_param(MTGPDC_PARAM_TABLE, block_num);
	make_texture(MTGPDC_PARAM_TABLE, d_texture, block_num);
	make_kernel_data(d_status, MTGPDC_PARAM_TABLE, block_num);
	make_uint32_random(d_status, num_data, block_num);

	//finalize
	CUDA_SAFE_CALL(cudaFree(d_status));
	CUDA_SAFE_CALL(cudaFree(d_texture[0]));
	CUDA_SAFE_CALL(cudaFree(d_texture[1]));
	CUDA_SAFE_CALL(cudaFree(d_texture[2]));
}
#endif

/* }}} */

__device__ void
initialize(const DevParameters *params, RNGState *state)
{
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	status_read(status, params->d_status, bid, tid);
	state->pos = pos_tbl[bid];
}

__device__ void
finalize(const DevParameters *params, RNGState *state)
{
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	status_write(params->d_status, status, bid, tid);
}

void
initialize_rng(DevParameters *params)
{
	int block_num = NUM_THREADS / BLOCKSIZE;
	CUDA_CHECK_ERROR(cudaMalloc((void**)&params->d_status, sizeof(mtgp32_kernel_status_t) * block_num));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_texture[0], sizeof(uint32_t) * block_num * TBL_SIZE));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_texture[1], sizeof(uint32_t) * block_num * TBL_SIZE));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_texture[2], sizeof(uint32_t) * block_num * TBL_SIZE));

	make_constant_param(MTGPDC_PARAM_TABLE, block_num);
	make_texture(MTGPDC_PARAM_TABLE, d_texture, block_num);
	make_kernel_data(params->d_status, MTGPDC_PARAM_TABLE, block_num);
}

void
destroy_rng(DevParameters *params)
{
	CUDA_CHECK_ERROR(cudaFree(params->d_status));
	CUDA_CHECK_ERROR(cudaFree(d_texture[0]));
	CUDA_CHECK_ERROR(cudaFree(d_texture[1]));
	CUDA_CHECK_ERROR(cudaFree(d_texture[2]));
}



} // namespace mtgp
