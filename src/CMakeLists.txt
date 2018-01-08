cmake_minimum_required(VERSION 2.8.4)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(MPI REQUIRED)
find_package(Boost COMPONENTS program_options mpi serialization system REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include_directories (SYSTEM ${Boost_INCLUDE_DIR})
link_directories (${Boost_LIBRARY_DIRS})

add_executable(rng_seq rng_seq.cpp MRG32k3a.c)
target_link_libraries(rng_seq ${Boost_LIBRARIES})

add_executable(rng_mpi rng_mpi.cpp MRG32k3a.c)
target_link_libraries(rng_mpi ${Boost_LIBRARIES} ${MPI_LIBRARIES})