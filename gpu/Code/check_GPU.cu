#include <stdio.h>

int main() {
  int devices;
  cudaGetDeviceCount(&devices);

  for (int d = 0; d < devices; ++d) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, d);
    int mp = p.multiProcessorCount, sp = 0;
    if (p.major == 2) {
      if (p.minor == 1) sp = 48;
      else sp = 32;
    } else if (p.major == 3) {
      sp = 192;
    } else if (p.major == 5) {
      sp = 128;
    }
    printf("Device %d: %s\n", d, p.name);
    printf(" -> multiprocessor count: %d\n", mp);
    printf(" -> stream processor count: %d (total %d)\n", sp, sp * mp);
    printf(" -> warp size: %d\n", p.warpSize);
    printf(" -> max threads per block: %d\n", p.maxThreadsPerBlock);
    printf(" -> max block dimensions: %d x %d x %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf(" -> max grid dimensions: %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    puts("");
  }
  return 0;
}