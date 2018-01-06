#include <iostream>
#include <stdlib.h>

#include "Generator.hpp"

using namespace std;

/**
* Generate a continuous raw binary stream
*/
int main(int argc, char * argv[]){
	// boost::mt19937 wrapper
	Random::Generator<int> g(0, 65535);
	while (true) {
		// Generate random number between 0 and 65535
		int x = g();

		//Print out the raw binary representation of the generated number
		cout << (char)((0xff00 & x) >> 8);
		cout << (char)(0x00ff & x);
	}

    return 0;
}