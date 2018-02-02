#ifndef ROUNDS
# define ROUNDS 64
#endif

#if ROUNDS > 64
# error max rounds for md5 is 64
#endif

namespace md5 {

enum {
	num_randoms_per_call = 4
};

struct RNGState
{
	int seed;
};

struct DevParameters
{
	int *seeds;
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

/* {{{ a lot of constants following here */
#define MD5_INIT_STATE_0 0x67452301
#define MD5_INIT_STATE_1 0xefcdab89
#define MD5_INIT_STATE_2 0x98badcfe
#define MD5_INIT_STATE_3 0x10325476
#define MD5_S11  7
#define MD5_S12 12
#define MD5_S13 17
#define MD5_S14 22
#define MD5_S21  5
#define MD5_S22  9
#define MD5_S23 14
#define MD5_S24 20
#define MD5_S31  4
#define MD5_S32 11
#define MD5_S33 16
#define MD5_S34 23
#define MD5_S41  6
#define MD5_S42 10
#define MD5_S43 15
#define MD5_S44 21
#define MD5_T01  0xd76aa478
#define MD5_T02  0xe8c7b756
#define MD5_T03  0x242070db
#define MD5_T04  0xc1bdceee
#define MD5_T05  0xf57c0faf
#define MD5_T06  0x4787c62a
#define MD5_T07  0xa8304613
#define MD5_T08  0xfd469501
#define MD5_T09  0x698098d8
#define MD5_T10  0x8b44f7af
#define MD5_T11  0xffff5bb1
#define MD5_T12  0x895cd7be
#define MD5_T13  0x6b901122
#define MD5_T14  0xfd987193
#define MD5_T15  0xa679438e
#define MD5_T16  0x49b40821
#define MD5_T17  0xf61e2562
#define MD5_T18  0xc040b340
#define MD5_T19  0x265e5a51
#define MD5_T20  0xe9b6c7aa
#define MD5_T21  0xd62f105d
#define MD5_T22  0x02441453
#define MD5_T23  0xd8a1e681
#define MD5_T24  0xe7d3fbc8
#define MD5_T25  0x21e1cde6
#define MD5_T26  0xc33707d6
#define MD5_T27  0xf4d50d87
#define MD5_T28  0x455a14ed
#define MD5_T29  0xa9e3e905
#define MD5_T30  0xfcefa3f8
#define MD5_T31  0x676f02d9
#define MD5_T32  0x8d2a4c8a
#define MD5_T33  0xfffa3942
#define MD5_T34  0x8771f681
#define MD5_T35  0x6d9d6122
#define MD5_T36  0xfde5380c
#define MD5_T37  0xa4beea44
#define MD5_T38  0x4bdecfa9
#define MD5_T39  0xf6bb4b60
#define MD5_T40  0xbebfbc70
#define MD5_T41  0x289b7ec6
#define MD5_T42  0xeaa127fa
#define MD5_T43  0xd4ef3085
#define MD5_T44  0x04881d05
#define MD5_T45  0xd9d4d039
#define MD5_T46  0xe6db99e5
#define MD5_T47  0x1fa27cf8
#define MD5_T48  0xc4ac5665
#define MD5_T49  0xf4292244
#define MD5_T50  0x432aff97
#define MD5_T51  0xab9423a7
#define MD5_T52  0xfc93a039
#define MD5_T53  0x655b59c3
#define MD5_T54  0x8f0ccc92
#define MD5_T55  0xffeff47d
#define MD5_T56  0x85845dd1
#define MD5_T57  0x6fa87e4f
#define MD5_T58  0xfe2ce6e0
#define MD5_T59  0xa3014314
#define MD5_T60  0x4e0811a1
#define MD5_T61  0xf7537e82
#define MD5_T62  0xbd3af235
#define MD5_T63  0x2ad7d2bb
#define MD5_T64  0xeb86d391
/* }}} */

__device__ unsigned int
rotate_left(unsigned int x, int n)
{
	return(x << n) | (x >>(32 - n));
}

__device__ void
FF(unsigned int& A, unsigned int B, unsigned int C, unsigned int D, unsigned int X, unsigned int S, unsigned int T)
{
	unsigned int F = (B & C) |(~B & D);
	A += F + X + T;
	A = rotate_left(A, S);
	A += B;
}

__device__ void
GG(unsigned int& A, unsigned int B, unsigned int C, unsigned int D, unsigned int X, unsigned int S, unsigned int T)
{
	unsigned int G = (B & D) |(C & ~D);
	A += G + X + T;
	A = rotate_left(A, S);
	A += B;
}

__device__ void
HH(unsigned int& A, unsigned int B, unsigned int C, unsigned int D, unsigned int X, unsigned int S, unsigned int T)
{
	unsigned int H = (B ^ C ^ D);
	A += H + X + T;
	A = rotate_left(A, S);
	A += B;
}

__device__ void
II(unsigned int& A, unsigned int B, unsigned int C, unsigned int D, unsigned int X, unsigned int S, unsigned int T)
{
	unsigned int I = (C ^(B | ~D));
	A += I + X + T;
	A = rotate_left(A, S);
	A += B;
}

__device__ void
hash_md5(unsigned int *data)
{
	unsigned int a = MD5_INIT_STATE_0;
	unsigned int b = MD5_INIT_STATE_1;
	unsigned int c = MD5_INIT_STATE_2;
	unsigned int d = MD5_INIT_STATE_3;

	unsigned int X[16];
	#pragma unroll
	for(int i = 0; i < 4; i++)
		X[i] = data[i];
	X[4] = 0x80000000;
	#pragma unroll
	for(int i = 5; i < 16; i++)
		X[i] = 0;

	switch(ROUNDS) {
	default:
	case 64: FF(a, b, c, d, X[ 0], MD5_S11, MD5_T01);
	case 63: FF(d, a, b, c, X[ 1], MD5_S12, MD5_T02);
	case 62: FF(c, d, a, b, X[ 2], MD5_S13, MD5_T03);
	case 61: FF(b, c, d, a, X[ 3], MD5_S14, MD5_T04);
	case 60: FF(a, b, c, d, X[ 4], MD5_S11, MD5_T05);
	case 59: FF(d, a, b, c, X[ 5], MD5_S12, MD5_T06);
	case 58: FF(c, d, a, b, X[ 6], MD5_S13, MD5_T07);
	case 57: FF(b, c, d, a, X[ 7], MD5_S14, MD5_T08);
	case 56: FF(a, b, c, d, X[ 8], MD5_S11, MD5_T09);
	case 55: FF(d, a, b, c, X[ 9], MD5_S12, MD5_T10);
	case 54: FF(c, d, a, b, X[10], MD5_S13, MD5_T11);
	case 53: FF(b, c, d, a, X[11], MD5_S14, MD5_T12);
	case 52: FF(a, b, c, d, X[12], MD5_S11, MD5_T13);
	case 51: FF(d, a, b, c, X[13], MD5_S12, MD5_T14);
	case 50: FF(c, d, a, b, X[14], MD5_S13, MD5_T15);
	case 49: FF(b, c, d, a, X[15], MD5_S14, MD5_T16);
	case 48: GG(a, b, c, d, X[ 1], MD5_S21, MD5_T17);
	case 47: GG(d, a, b, c, X[ 6], MD5_S22, MD5_T18);
	case 46: GG(c, d, a, b, X[11], MD5_S23, MD5_T19);
	case 45: GG(b, c, d, a, X[ 0], MD5_S24, MD5_T20);
	case 44: GG(a, b, c, d, X[ 5], MD5_S21, MD5_T21);
	case 43: GG(d, a, b, c, X[10], MD5_S22, MD5_T22);
	case 42: GG(c, d, a, b, X[15], MD5_S23, MD5_T23);
	case 41: GG(b, c, d, a, X[ 4], MD5_S24, MD5_T24);
	case 40: GG(a, b, c, d, X[ 9], MD5_S21, MD5_T25);
	case 39: GG(d, a, b, c, X[14], MD5_S22, MD5_T26);
	case 38: GG(c, d, a, b, X[ 3], MD5_S23, MD5_T27);
	case 37: GG(b, c, d, a, X[ 8], MD5_S24, MD5_T28);
	case 36: GG(a, b, c, d, X[13], MD5_S21, MD5_T29);
	case 35: GG(d, a, b, c, X[ 2], MD5_S22, MD5_T30);
	case 34: GG(c, d, a, b, X[ 7], MD5_S23, MD5_T31);
	case 33: GG(b, c, d, a, X[12], MD5_S24, MD5_T32);
	case 32: HH(a, b, c, d, X[ 5], MD5_S31, MD5_T33);
	case 31: HH(d, a, b, c, X[ 8], MD5_S32, MD5_T34);
	case 30: HH(c, d, a, b, X[11], MD5_S33, MD5_T35);
	case 29: HH(b, c, d, a, X[14], MD5_S34, MD5_T36);
	case 28: HH(a, b, c, d, X[ 1], MD5_S31, MD5_T37);
	case 27: HH(d, a, b, c, X[ 4], MD5_S32, MD5_T38);
	case 26: HH(c, d, a, b, X[ 7], MD5_S33, MD5_T39);
	case 25: HH(b, c, d, a, X[10], MD5_S34, MD5_T40);
	case 24: HH(a, b, c, d, X[13], MD5_S31, MD5_T41);
	case 23: HH(d, a, b, c, X[ 0], MD5_S32, MD5_T42);
	case 22: HH(c, d, a, b, X[ 3], MD5_S33, MD5_T43);
	case 21: HH(b, c, d, a, X[ 6], MD5_S34, MD5_T44);
	case 20: HH(a, b, c, d, X[ 9], MD5_S31, MD5_T45);
	case 19: HH(d, a, b, c, X[12], MD5_S32, MD5_T46);
	case 18: HH(c, d, a, b, X[15], MD5_S33, MD5_T47);
	case 17: HH(b, c, d, a, X[ 2], MD5_S34, MD5_T48);
	case 16: II(a, b, c, d, X[ 0], MD5_S41, MD5_T49);
	case 15: II(d, a, b, c, X[ 7], MD5_S42, MD5_T50);
	case 14: II(c, d, a, b, X[14], MD5_S43, MD5_T51);
	case 13: II(b, c, d, a, X[ 5], MD5_S44, MD5_T52);
	case 12: II(a, b, c, d, X[12], MD5_S41, MD5_T53);
	case 11: II(d, a, b, c, X[ 3], MD5_S42, MD5_T54);
	case 10: II(c, d, a, b, X[10], MD5_S43, MD5_T55);
	case  9: II(b, c, d, a, X[ 1], MD5_S44, MD5_T56);
	case  8: II(a, b, c, d, X[ 8], MD5_S41, MD5_T57);
	case  7: II(d, a, b, c, X[15], MD5_S42, MD5_T58);
	case  6: II(c, d, a, b, X[ 6], MD5_S43, MD5_T59);
	case  5: II(b, c, d, a, X[13], MD5_S44, MD5_T60);
	case  4: II(a, b, c, d, X[ 4], MD5_S41, MD5_T61);
	case  3: II(d, a, b, c, X[11], MD5_S42, MD5_T62);
	case  2: II(c, d, a, b, X[ 2], MD5_S43, MD5_T63);
	case  1: II(b, c, d, a, X[ 9], MD5_S44, MD5_T64);
	};

	data[0] = a;
	data[1] = b;
	data[2] = c;
	data[3] = d;
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
		unsigned int arg[4] = {
			idx, state->seed++, 0, 0
		};

		hash_md5(arg);

		#pragma unroll
		for(int i = 0; i < md5::num_randoms_per_call && i < num_randoms; i++) {
			*out = arg[i];
			out += stride;
			num_randoms--;
		}
	}
}

} //namespace md5
