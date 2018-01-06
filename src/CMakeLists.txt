#
# [CMake](http://www.cmake.org/) Configuration
#
cmake_minimum_required (VERSION 2.8.4)
project (TutorialMatrixSTL CXX)


#SET(CMAKE_VERBOSE_MAKEFILE on )
SET(CMAKE_BUILD_TYPE       Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#====== [Boost](http://www.boost.org/) ======
# Note: uBlas is part of the 'numeric' library and is NOT a compiled library
find_package(Boost COMPONENTS random program_options timer system REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
list(APPEND EXTRALIBS ${Boost_LIBRARIES})


# Sources of the main executable
set(prng_HEADERS Generator.hpp)

add_executable(prng main.cpp ${prng_HEADERS})
target_link_libraries(prng ${EXTRALIBS})
