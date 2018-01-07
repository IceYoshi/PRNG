# PRNG

In order to test the quality of our PRNG implementation using dieharder:
```
./rng_seq | dieharder -g 200 -a

mpirun -np 4 rng_mpi | dieharder -g 200 -a
```

Useful dieharder options:

Replace -a with -d x in order to only execute a particular test.

Specify -m x in order to test with a bigger pool of random numbers. This will also take x-times longer to run.