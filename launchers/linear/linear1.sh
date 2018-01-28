#!/bin/bash -l
#SBATCH -J PRNGBenchmark
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=mike.pereira.001@student.uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

module purge
module load toolchain/foss/2017a
module load devel/Boost/1.65.1-foss-2017a

cd ~/PRNG/build

# Extract and output real time (in seconds)
TIMEFORMAT=%R

echo "Start of linear test with g=1"

time ./rng_seq -g 1 -n 100 &> /dev/null
time ./rng_seq -g 1 -n 1000 &> /dev/null
time ./rng_seq -g 1 -n 10000 &> /dev/null
time ./rng_seq -g 1 -n 100000 &> /dev/null
time ./rng_seq -g 1 -n 1000000 &> /dev/null
time ./rng_seq -g 1 -n 10000000 &> /dev/null
time ./rng_seq -g 1 -n 100000000 &> /dev/null
time ./rng_seq -g 1 -n 1000000000 &> /dev/null
time ./rng_seq -g 1 -n 10000000000 &> /dev/null
time ./rng_seq -g 1 -n 100000000000 &> /dev/null

echo "End of benchmarking"
