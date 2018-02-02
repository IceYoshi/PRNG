#ifndef  __UTIL_HPP__
#define  __UTIL_HPP__

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK_ERROR(a) \
	do { \
		cudaError r = a; \
		if (r != cudaSuccess) \
			fprintf(stderr, "CUDA ERROR: line %d <%s> error %d\n", __LINE__, #a, r); \
	} while(0)


inline void
choose_device()
{
	int num_devices, use_device;
	cudaDeviceProp device_prop;

	cudaGetDeviceCount(&num_devices);

	const char *device_pick = getenv("CUDA_DEVICE");
	use_device = 0;
	if(device_pick) {
		errno = 0;
		use_device = strtol(device_pick, NULL, 10);
		if(errno || use_device >= num_devices) {
			fprintf(stderr, "invalid device number\n");
			exit(EXIT_FAILURE);
		}
	}

	cudaGetDeviceProperties(&device_prop, use_device);
	fprintf(stderr, "using dev %d: %s\n", use_device, device_prop.name);

	cudaSetDevice(use_device);
}


template<int x> struct next_or_equal_power_of_two { enum { value = x }; };
template<> struct next_or_equal_power_of_two<3>   { enum { value = 4 }; };
template<> struct next_or_equal_power_of_two<5>   { enum { value = 8 }; };
template<> struct next_or_equal_power_of_two<6>   { enum { value = 8 }; };
template<> struct next_or_equal_power_of_two<7>   { enum { value = 8 }; };

#endif  /*__UTIL_HPP__*/
