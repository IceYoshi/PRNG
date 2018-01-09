# PRNG

**1. In order to test the quality of our PRNG implementation using dieharder:**
```
./rng_seq | dieharder -g 200 -a

mpirun -np <numOfProcesses> rng_mpi | dieharder -g 200 -a
```

Useful dieharder options (`x` stands for any number):
- Replace `-a` with `-d x` in order to only execute a particular test. `-d -1` in order to list all available tests.
- Specify `-m x` in order to test with a bigger pool of random numbers. Note that the test will also take `x`-times longer to run.

**2. For the NIST test suite:**

- Prepare a binary file in advance, e.g. `./rng_seq -n <bitStreamLength> > data/random.dat`
- Execute `./assess <bitStreamLength>` and follow the instructions, e.g:

    *Input file(0) -> data/random.dat -> All(1) -> Default(0) -> Bitstreams(10) -> Binary(1)*

After successful completion, results can be found under *experiments/AlgorithmTesting/finalAnalysisReport.txt*
