#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <random>		/* mt19937 */

#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <boost/program_options.hpp>

#include "MRG32k3a.c"

double MRG32k3a(void);
void setMRGSeed(int);

namespace po = boost::program_options;
namespace mpi = boost::mpi;
using namespace std;

int id = 0; // MPI id for the current process (set global to be used in xprintf)

int main(int argc, char *argv[]) {
	mpi::environment env;
	mpi::communicator world;

	id = world.rank();

	unsigned long n;// Stream length
	unsigned int g;	// Generator type
	unsigned int m; // Applied method for parallelism
	bool h = false;
	time_t seed = time(NULL);
	mt19937 mt;

	if (id == 0) {
		po::options_description desc("Available options");
		// supported options
		desc.add_options()
			("help,h", "Display this help message")
			("streamlength,n", po::value<unsigned long>()->default_value(0), "Number of bytes to be generated. Set to 0 for infinite execution.")
			("generator,g", po::value<unsigned int>()->default_value(0), "Type of generator. 0 for rand(), 1 for mt19937, 2 for MRG32k3a")
			("mode,m", po::value<unsigned int>()->default_value(0), "Method for parallelism. 0 for different parameter sets, 1 for block-splitting, 2 for leap-frogging")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm); //assign the variables (if they were specified)

		if (vm.count("help")) {
			cout << desc << endl;
			h = true;
		}

		n = vm["streamlength"].as<unsigned long>();
		g = vm["generator"].as<unsigned int>();
		m = vm["mode"].as<unsigned int>();
	}

	world.barrier();

	mpi::broadcast(world, n, 0);
	mpi::broadcast(world, g, 0);
	mpi::broadcast(world, m, 0);
	mpi::broadcast(world, h, 0);

	if (h) return 1;

	//-----------------------------------------------

	// Initialize rng and set seed depending on the selected mode
	if (m == 0) {
		srand(seed + id);
		mt = mt19937(seed + id);
		setMRGSeed(seed + id);
	}
	else if (m == 1) {
		srand(seed);
		mt = mt19937(seed);
		setMRGSeed(seed);
		int skipAhead = ceil((id * n * 1.0) / world.size());
		for (int i = 0; i < skipAhead; i++) {
			if (g == 0) rand(); else if (g == 1) mt(); else MRG32k3a();
		}
	}
	else if (m == 2) {
		srand(seed);
		mt = mt19937(seed);
		setMRGSeed(seed);
		for (int i = 0; i < id; i++) {
			if (g == 0) rand(); else if (g == 1) mt(); else MRG32k3a();
		}
	}
	else {
		cout << "Error: Undefined mode " << m << endl;
		return 3;
	}

	unsigned long count = 0;
	unsigned long x;
	while (n == 0 || count++ < n) {
		// Generate random number between 0 and 2^32 - 1
		if (g == 0) {		// rand()
			x = rand();
		}
		else if (g == 1) {	// mt19937
			x = mt();
		}
		else if (g == 2) {	// MRG32k3a
			x = (unsigned long)( MRG32k3a() * (1UL << 32) );
		}
		else {
			cout << "Error: Undefined generator " << g << endl;
			return 2;
		}

		//Print out the raw binary representation of the generated number
		cout << (char)((0xff000000 & x) >> 24);
		cout << (char)((0x00ff0000 & x) >> 16);
		cout << (char)((0x0000ff00 & x) >> 8);
		cout << (char)(0x000000ff & x);

		if (m == 2) {
			for (int i = 0; i < world.size(); i++) {
				if (g == 0) rand(); else mt();
			}
		}
	}
	//-----------------------------------------------

	return 0;
}