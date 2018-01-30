#!/bin/bash -l
#SBATCH -J PRNGDieharder
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

srun -n 56 ./rng_mpi -g 1 -m 0 | dieharder -g 200 -a