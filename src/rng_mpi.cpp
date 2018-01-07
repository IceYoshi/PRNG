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

	xprintf("Test output\n");

	return 0;
}