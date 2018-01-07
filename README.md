# PRNG

In order to test the quality of our PRNG implementation using dieharder:
```
./rng_seq | dieharder -g 200 -a

mpirun -np <numOfProcesses> rng_mpi | dieharder -g 200 -a

srun -n <numOfProcesses> rng_mpi | dieharder -g 200 -a
```

Useful dieharder options (`x` stands for any number):

Replace `-a` with `-d x` in order to only execute a particular test. `-d -1` in order to list all available tests.

Specify `-m x` in order to test with a bigger pool of random numbers. Note that the test will also take `x`-times longer to run.
