#ifndef  __SEED_HPP__
#define  __SEED_HPP__

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>


/* utility function to read in the first 'size' bytes into specified memory
 * region
 */
inline void
seed_memory(void *mem, size_t size, const char *path = "real_randoms")
{
	static size_t file_offset = 0;
	int file = open(path, O_RDONLY);
	struct stat sb;
	int s, offset;

	if(file < 0) {
		perror("error opening file");
		exit(EXIT_FAILURE);
	}
	if(fstat(file, &sb) < 0) {
		perror("stat() failed");
		exit(EXIT_FAILURE);
	}
	if(sb.st_size < file_offset + size) {
		std::cerr << "random data file too small for seeding" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(lseek(file, file_offset, SEEK_SET)) {
		perror("setting file offset");
		exit(EXIT_FAILURE);
	}
	
	offset = 0;
	while(size > 0) {
		s = read(file, (char *)mem + offset, size);
		if(s < 0) {
			perror("error reading");
			exit(EXIT_FAILURE);
		}
		offset += s;
		size -= s;
	}

	close(file);
	file_offset += size;
}

inline unsigned int *
allocate_seeded_device_memory(size_t size)
{
	unsigned int *seeds, *seeds_dev;
	CUDA_CHECK_ERROR(cudaMalloc(&seeds_dev, size));
	CUDA_CHECK_ERROR(cudaMallocHost(&seeds, size));
	seed_memory(seeds, size);
	CUDA_CHECK_ERROR(cudaMemcpy(seeds_dev, seeds, size, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaFreeHost(seeds));

	return seeds_dev;
}


#endif  /*__SEED_HPP__*/
