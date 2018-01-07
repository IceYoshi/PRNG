#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit  */
#include <stdarg.h>    /* for va_{list,args... */
#include <string>
#include <iostream>

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "Generator.hpp"

namespace mpi = boost::mpi;
using namespace std;

int id = 0; // MPI id for the current process (set global to be used in xprintf)

void xprintf(std::string format, ...) {
	va_list args;
	va_start(args, format);
	printf("[Node %i] ", id);
	vprintf(format.c_str(), args);
	fflush(stdout);
}

int main(int argc, char *argv[]) {
	mpi::environment env;
	mpi::communicator world;

	id = world.rank();

	world.barrier();

	//-----------------------------------------------
	unsigned int bitSize = 16;

	// boost::mt19937 wrapper
	Random::Generator<int> g(0, (1 << bitSize) - 1, id);

	// Initialize seed for rand()
	//srand(time(NULL));

	while (true) {
		// Generate random number between 0 and 2^16 - 1 (25535)
		//int x = rand() % (1 << bitSize);
		int x = g();

		//Print out the raw binary representation of the generated number
		cout << (char)((0xff00 & x) >> 8);
		cout << (char)(0x00ff & x);
	}
	//-----------------------------------------------

	return 0;
}