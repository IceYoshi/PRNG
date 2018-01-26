#!/bin/bash -l
#SBATCH -J PRNGBenchmark1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=mike.pereira.001@student.uni.lu
#SBATCH -N 2
#SBATCH --ntasks-per-node=28
#SBATCH --time=0-02:00:00
#SBATCH -p batch
#SBATCH --qos=qos-batch

module purge
module load toolchain/foss/2017a
module load devel/Boost/1.65.1-foss-2017a

cd ~/PRNG/build

echo "Start of mode test with m=0 (cont.)"

time mpirun -np 28 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 30 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 35 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 40 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 45 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 50 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 55 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null
time mpirun -np 56 ./rng_mpi -g 1 -m 0 -n 100000000000 &> /dev/null

echo "End of benchmarking"
