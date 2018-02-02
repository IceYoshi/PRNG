# See COPYING for copyright and license details

# this file configures the benchmark suite


# add new RNGs here
########################################################################


RANDOM_NUMBER_GENERATORS = \
						   md5 \
						   tea \
						   drand48gpu \
						   park_miller \
						   tt800 \
						   lfsr113 \
						   combinedLCGTaus \
						   kiss07 \
						   mtgp \
						   ranecu \
						   empty \
						   precompute

#RANDOM_NUMBER_GENERATORS += \
#   $(foreach num_rounds,$(shell seq 10 4 63),md5.ROUNDS-$(num_rounds)) \
#   $(foreach num_rounds,$(shell seq 1 31),tea.ROUNDS-$(num_rounds)) \

# configuration for the cuda stuff
########################################################################

NVCC             = $(CUDA_BIN)/nvcc
CUDA_BIN         = /mnt/gaiagpfs/users/homedirs/ltrestioreanu/mprng/cuda/bin
CUDA_INCLUDES    = -I/mnt/gaiagpfs/users/homedirs/ltrestioreanu/mprng/cuda/include
CUDA_LIBS        = -lcudart
CUDA_LIB_PATHS   = -L/mnt/gaiagpfs/users/homedirs/ltrestioreanu/mprng/cuda/lib64
NVCCFLAGS        = -arch sm_30

# config for the dieharder test
########################################################################

DIE_HARDER_BIN   = /usr/bin/dieharder
DIE_HARDER_FLAGS = -g 200 -a -c ',' -D 33272

# config for testu01 test
########################################################################

TESTU01_BIN_DIR = testu01_binaries


# config for the raw performance benchmark
########################################################################

# program calls
RAW_PERFORMANCE_NUM_ROUNDS = 10
# num of kernel calls
RAW_PERFORMANCE_NUM_KERNEL_CALLS = 10

# must be power of 2
RAW_PERFORMANCE_NUM_THREADS = 16384

# random numbers calculated in each thread in each kernel call
RAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD = 2048

# resulting amount of random numbers per program launch =
#  kernel_calls * num_threads * num_randoms_per_thread

#
########################################################################

# program calls
ASIAN_OPTIONS_NUM_ROUNDS = 10

# must be power of 2
ASIAN_OPTIONS_NUM_THREADS = 16384

# number of kernel calls
ASIAN_OPTIONS_NUM_KERNEL_CALLS = 10

# ???
ASIAN_OPTIONS_NUM_RUNS = 16


# directories used for the testsuite. you should never need to edit those
########################################################################

TEST_DIR         = tests
RESULT_DIR       = results
RND_MAKEFILE_DIR = rng_makefiles
RND_DIR          = rng_implementations
RND_BIN          = rng_bin
STUB_DIR         = stubs
BUILD_LOGS       = build_logs

