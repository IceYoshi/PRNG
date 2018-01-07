#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <boost/program_options.hpp>

#include "Generator.hpp"

namespace po = boost::program_options;
using namespace std;

/**
* Generate a continuous raw binary stream
*/
int main(int argc, char * argv[]){
	po::options_description desc("Available options");
	// supported options
	desc.add_options()
		("help,h", "Display this help message")
		("streamlength,n", po::value<unsigned long>()->default_value(0), "Number of bytes to be generated. Set to 0 for infinite execution.")
		("generator,g", po::value<unsigned int>()->default_value(0), "Type of generator. 0 for rand(), 1 for mt19937")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm); //assign the variables (if they were specified)

	if (vm.count("help")) {
		cout << desc << endl;
		return 1;
	}

	unsigned long n = vm["streamlength"].as<unsigned long>();
	unsigned int g = vm["generator"].as<unsigned int>();

	unsigned int bitSize = 16;

	// Initialize seed for rand()
	srand(time(NULL));

	// boost::mt19937 wrapper
	Random::Generator<int> mt(0, (1 << bitSize) - 1);

	unsigned long count = 0;
	int x;
	while (n == 0 || count++ < n) {
		// Generate random number between 0 and 2^16 - 1 (25535)
		switch (g) {
		case 0: // rand()
			x = rand() % (1 << bitSize);
			break;
		case 1:	// mt19937
			x = mt();
			break;
		default:
			cout << "Error: Undefined generator " << g << endl;
			return 2;
		}
		
		//Print out the raw binary representation of the generated number
		cout << (char)((0xff00 & x) >> 8);
		cout << (char)(0x00ff & x);
	}

    return 0;
}