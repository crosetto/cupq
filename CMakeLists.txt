cmake_minimum_required( VERSION 3.9 )
project( dijkstra CUDA CXX )

set ( CMAKE_CXX_RELEASE_FLAGS ${CMAKE_CXX_RELEASE_FLAGS} -O3)
set ( CMAKE_CUDA_RELEASE_FLAGS ${CMAKE_CXX_RELEASE_FLAGS} -O3)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set( CMAKE_CXX_STANDARD 14 )
  set( CMAKE_CXX_STANDARD_REQUIRED TRUE )
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set( CMAKE_CUDA_STANDARD 14 )
  set( CMAKE_CUDA_STANADARD_REQUIRED TRUE )
endif()

## LIBRARY

set( COMPUTE_CAPABILITY "50" CACHE STRING "set compute capability (e.g. \"50\" for 5.0)")

#using unified memory starting from Pascal generation GPUs
if ("${COMPUTE_CAPABILITY}" STRGREATER "60")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUNIFMEM")
endif("${COMPUTE_CAPABILITY}" STRGREATER "60")

add_library(dijkstra_simple_gpu SHARED examples/dijkstra_simple/dijkstra_simple.cu)
add_library(dijkstra_simple_cpu SHARED examples/dijkstra_simple/dijkstra_simple.cpp)

set_target_properties(dijkstra_simple_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

target_include_directories(dijkstra_simple_gpu INTERFACE src)
target_include_directories(dijkstra_simple_cpu INTERFACE src)

add_library(compiler_info INTERFACE)

set(cuda_lang "$<COMPILE_LANGUAGE:CUDA>")

target_include_directories(compiler_info INTERFACE
					 $<BUILD_INTERFACE: ${CMAKE_CURRENT_SOURCE_DIR}/compiler_info>
					 $<INSTALL_INTERFACE:include/example/compiler_info>)

target_compile_options(compiler_info INTERFACE
				     $<${cuda_lang}:$<BUILD_INTERFACE:
				     -gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY} -lineinfo --ptxas-options=-v --default-stream per-thread >>
				     )

target_link_libraries(dijkstra_simple_gpu PUBLIC compiler_info)

##EXAMPLES

add_executable(dijkstra_simple_exe examples/dijkstra_simple/main.cpp)
target_link_libraries(dijkstra_simple_exe PRIVATE dijkstra_simple_gpu dijkstra_simple_cpu)

##dijkstra example configuration options

option(USE_CACHE "whether to use caches" OFF)
option(COUNT_HITS "counting the number of cache hits" OFF)
option(PATH_LENGTH "printing out the average shortest path length" OFF)
option(TIMINGS "whether to print out timing information" OFF)
option(CPU_COMPARISON "whether to compare timing with CPU" OFF)
option(VERBOSE "verbose execution (only for very small nets)" OFF)
option(WRITE "write out result in a text file" OFF)

if (USE_CACHE)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -UDISABLE_CACHE")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -UDISABLE_CACHE")
else(USE_CACHE)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDISABLE_CACHE")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDISABLE_CACHE")
endif(USE_CACHE)

if (COUNT_HITS)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCOUNT_HITS")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCOUNT_HITS")
endif(COUNT_HITS)

if (PATH_LENGTH)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSTATISTICS")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTATISTICS")
endif(PATH_LENGTH)

if (TIMINGS)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DTIMINGS")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIMINGS")
endif(TIMINGS)

if (CPU_COMPARISON)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCPU_COMPARISON")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_COMPARISON")
endif(CPU_COMPARISON)

if (VERBOSE)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DVERBOSE")
   set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} -DVERBOSE")
endif(VERBOSE)

if (WRITE)
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWRITE")
   set(CMAKE_CXX_FLAGS "${CMAKE_CUDA_FLAGS} -DWRITE")
endif(WRITE)

##TPLs
find_package(Boost REQUIRED COMPONENTS timer program_options)
if (Boost_FOUND)

   target_include_directories(dijkstra_simple_exe PRIVATE ${Boost_INCLUDE_DIRS})
   target_link_libraries(dijkstra_simple_exe PRIVATE ${Boost_LIBRARIES})
endif(Boost_FOUND)

##TESTs
add_executable(reverse tests/test_reverse.cu)
target_link_libraries(reverse PUBLIC compiler_info PRIVATE ${Boost_LIBRARIES})
add_test(NAME reverse COMMAND reverse)

add_custom_target(download_graphs BYPRODUCTS USA-road-d.NY.gr.gz USA-road-d.NY.co.gz USA-road-d.BAY.gr.gz USA-road-d.BAY.co.gz
				  COMMAND wget http://users.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz; COMMAND gunzip USA-road-d.NY.gr.gz
                                  COMMAND wget http://users.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.co.gz; COMMAND gunzip USA-road-d.NY.co.gz;
                                  COMMAND wget http://users.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.BAY.gr.gz; COMMAND gunzip USA-road-d.BAY.gr.gz;
                                  COMMAND wget http://users.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.BAY.co.gz; COMMAND gunzip USA-road-d.BAY.co.gz;
                                  WORKING_DIRECTORY ${CMAKE_CURRENT_BUILD_DIR})
