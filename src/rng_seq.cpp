#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "Generator.hpp"

using namespace std;

/**
* Generate a continuous raw binary stream
*/
int main(int argc, char * argv[]){
	unsigned int bitSize = 16;

	// boost::mt19937 wrapper
	//Random::Generator<int> g(0, (1 << bitSize) - 1);

	// Initialize seed for rand()
	srand(time(NULL));

	while (true) {
		// Generate random number between 0 and 2^16 - 1 (25535)
		int x = rand() % (1 << bitSize);
		//int x = g();
		
		//Print out the raw binary representation of the generated number
		cout << (char)((0xff00 & x) >> 8);
		cout << (char)(0x00ff & x);
	}

    return 0;
}