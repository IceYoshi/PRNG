#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <errno.h>

using namespace std;

int
main(int argc, char **argv)
{
	int num_devices, use_device;
	cudaDeviceProp device_prop;

	cudaGetDeviceCount(&num_devices);

	printf("number of devices: %d\n", num_devices);

	const char *device_pick = getenv("CUDA_DEVICE");
	use_device = 0;
	if(device_pick) {
		errno = 0;
		use_device = strtol(device_pick, NULL, 10);
		if(errno || use_device >= num_devices) {
			printf("invalid device number\n");
			exit(EXIT_FAILURE);
		}
	}

	cudaGetDeviceProperties(&device_prop, use_device);
	fprintf(stdout, "using dev %d: %s\n", use_device, device_prop.name);

	return 0;
}

