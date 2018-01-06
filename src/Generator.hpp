#ifndef __GENERATOR_HPP
#define __GENERATOR_HPP

#include <boost/random.hpp>

namespace Random {
	template<class T,
		class IntType = int,
		class Engine = boost::mt19937>
		class Generator {
		boost::random::uniform_int_distribution<> _dist; /**< type of distribution */
		Engine _rng; /** Random Number Generator */
		public:
			Generator(const IntType & min = IntType(0), const IntType & max = IntType(100)) :
				_dist(min, max), _rng(std::time(nullptr)) {}
			virtual ~Generator() {}
			IntType min() const { return _dist.min(); }
			IntType max() const { return _dist.max(); }
			T operator()() {
				return T(_dist(_rng));
			}
	};      // ============ end class Generator =============


}; // namespace Random

#endif  // _GENERATOR_HPP