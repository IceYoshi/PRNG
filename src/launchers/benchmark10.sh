#!/bin/bash -l
#SBATCH -J PRNGBenchmark1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=mike.pereira.001@student.uni.lu
#SBATCH -N 1
#SBATCH --ntasks-per-node=28
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

module purge
module load toolchain/foss
module load devel/Boost/1.65.1-foss-2017a

cd ~/PRNG/build

echo "Start of benchmarking with g=1"

time mpirun -np   1 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np   5 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np  10 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np  15 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np  20 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np  25 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np  28 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null

echo "End of benchmarking"
