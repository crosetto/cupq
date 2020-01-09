# CUPQ: a CUDA implementation of a Priority Queue applied to the many-to-many shortest path problem

This library implements a priority queue in CUDA, the goal (besides showing the data structure and its implementation) is to compare it with the CPU implementations
 (from Boost or the STL) on a Dijkstra algorithm benchmark. I used for comparison the GPUs in my laptops (a Maxwell M1200 and GeForce GTX 1050 Mobile GPUs).
We assume it's fair to compare them with a single core of a high-end multicore CPU (in this case i7-7820HQ CPU). In fact a gaming GPU such as the GTX 1080 would have approximately 
4 times the hardware and memory bandwidth compared to these laptop GPUs, and also the prices of the GTX 1080 and the multicore processor i7-7820HQ are approximately matching (around 400$).

For benchmarking purposes we use a publicly available road networks database ```http://users.diag.uniroma1.it/challenge9/download.shtml```.
The problem we are addressing is the simultaneous computation of the shortest paths from thousands of origins to all the nodes of the graph.

## Quick start

run CMake to configure the project
```shell
> mkdir build;
> cd build; cmake -DCOMPUTE_CAPABILITY=61 -DTIMINGS -DCPU_COMPARISON -DCMAKE_BUILD_TYPE=RELEASE ..
```
where you have to input the compute capability of your NVidia graphic card without the dot (for the GTX 1050 Mobile the compute capability is 6.1).

Download and unzip a couple of input graphs to run the examples
```shell
> make download_grahs
```

build the examples
```shell
> make -j
```

Run the shortest paths benchmark
```shell
> ./dijkstra_simple_exe -g USA-road-d.NY
```

This benchmark will select randomly origins and destinations and execute the algorithm. You can set the number of origins/destinations by editing the file examples/dijkstra_simple/main.cpp.
You can also tune parameters in examples/dijkstra_simple/dijkstra_simple.cu such as the number of streams (useful to overlap the kernels execution and data transfer), and the chunk size (the number of origins consumed at each kernel launch).

## Heap Data Structure

The basic idea of this CUDA implementation of the many-to-many shortest path problem consists in assigning a heap-based priority queue to each CUDA *warp*.
The heap is a balanced tree data structure which satisfies the *heap property*: the value of the parent node is smaller than or equal to the value of the children.

Actually we don't really use a regular heap: we replace the nodes in the heap with sorted arrays
of 32 items, where an item is a pair containing a value (32 bit floating point number representing the distance from the origin) and a key (32 bit integer identifiyng the node in the graph).
These arrays of items are sorted in increasing order with respect to the item's values,
and the value of the smallest item in each array is satisfying a *heap-like property*: the first element of
each array has smaller cost than the first elements of it's children's arrays.
We call such first elements the *top* elements of each array.

If you are confused after this explanation look at the clarifying picture below:

![alt text](doc/fig_heap.png)

The two basic operations performed in a heap are:
  - *pop*: extraction of the minumum element (the top item in the root array), then in order to fill the empty slot the top element of the last node of the heap is moved to the top position in the root array. The change is then propagated down the tree until the heap property is recovered;
  - *insert*: a new element is inserted at the bottom of the heap, adding a new child if necessary (but keeping the tree balanced), otherwise filling an empty slot in an existing array (and maintaining the order whithin the array), then the change is propagated up until the heap property is recovered.

Each thread in the warp is *responsible* for one value in the arrays. To keep the array sorted, when doing operations like extracting the minimum, or inserting a new element, 
the threads have to compare and shuffle their own value with each other, and that's
roughly where the GPU ballot, bfind, and shuffle instructions come into play (in ![src/device/node.h](src/device/node.h) ).

In particular say we want to propagate down an item, whithin a pop, and evict another item from the current node which will be propaagted further down.
We first take the minimum of the children by comparing their *top* items weights. Once we have selected the min child, we compare the input item with the top item of the min child array.
If the input item's weight is smaller than the top weight in the min child,
then the heap property is already satisfied,
so we just return the input element. Otherwise, if the input element's weights is not the smallest one in the min child array,
we create a bit mask in a 32 bits register using the GPU ballot instruction. Each thread compares its own item's cost in the min child array with the cost of the input item,
and sets its correspondig bit in the mask to 1 if it is smaller, to 0 otherwise. Notice that there is at least one bit set to 0 in the mask,
because the input item is not the smallest one whithin the child array.
We get a mask like 00000001111111111111111111111111. We add 1 to this register, to get
00000010000000000000000000000000 (this ensures that there is at least one bit set in the register).
The location of the first nonzero bit is the location where the given item must be inserted, and we use the *bfind* PTX intrinsic to get it.
Once we identifiy the location of the fist nonzero bit, we use the shuffle instruction to shift up all the items in the array that have a smaller cost than the input item.
The evicted item is the smallest element in the child array, which is inserted in the parent node. The new
top item of the min child array is the one to be further compared with its children to restore the heap property.

When propagating up, during the *insert* procedure, we use a similar strategy. Selecting the parent node is just an algebraic operation on the child index. We start from a leaf node,
where we insert the input item,
and traverse the heap upward. At each hop we store in a stack an ordinal identifying the child node, and we continue until we find a node in which the *top* item has a smaller cost than
the input one. At this point we insert the item in the array of the node we found, in a similar way as for the *pop* procedure, just replacing the shift-up with a shift-down, and evicting the largest element instead of the smallest one. We then follow downward the path we recorded by unwinding the stack, inserting the propagated item and evicting the largest one
at each hop, until we reach the leaf.

With this idea we exploit at best the memory bandwidth, 
since all threads are accessing their own element in the arrays concurrently, and they communicate through the shuffle instruction. 
The operations of the Dijkstra algorithm are instead redundantly computed by all the cores in the warp.

The simplest implementation of a heap data structure would be a binary heap, as the one shown in the picture above. However increasing the heap *arity* reduces the number of hops performed to reach a given node, thus might reduce expensive memory accesses.
We choose an arity of 32, and this choice will be motivated below. We will consider thus a 32-ways heap in which each node contains 32 values, so that accessing a heap
smaller than 1056 items only requires at most one hop.

We keep the top element of each array in the fast *shared memory*, so that
each cuda thread in a warp will be able to readly access his own element of each array, and the smallest element of each array.

For this reason having a ``fat'' heap
turns out to be more efficient than a classical binary heap. In fact when we do a pop operation,
and propagate the change down the heap, we compare the cost of the parent's top item (cached in shared memory) with the top items of the children (also in shared memory),
in order to select which child array to access. Having larger arity increases the number of comparisons, but it decreases the height of the heap,
which means trading expensive local memory accesses for cheap shared memory ones. A natural choice for the arity is 32,
since we can take the minimum of 32 numbers whithin a warp in parallel, by calling 5 times the
shuffle instruction, as shown below with an example.

![alt text](doc/fig_min.png)

In the picture above we take 32 registers. We compare at each level pairs of elements, and whenever they are not in increasing order we copy the second element into the first one.
After repeating this 5 times we have the smallest element in the leftmost position.

With the model described here each warp accesses its own priority queue, which is sitting on the
threads local
memory, so that multiple warps will execute different shortest path algorithms concurrently, computing
eventually the solution to the many-to-all shortest path problem.

## Results

Computing the Shortest Path typically involves
little computation (i.e. arithmetic operations), and a large amount of memory accesses. Algorithms with such behaviour are called
*memory-bound* algorithms, since their performance is bound by the speed of the memory accesses, as opposed to *compute-bound* algorithms which are limited by the
speed of the Arithmetic Logic Units (ALU) or Floating Point Units (FPU). For memory bound
 applications the computation time is often completely hidden by the latency of memory accesses, thus
improving the efficiency of the arithmetic operations (e.g. by choosing processors with higher clock frequency, vectorizing floating point operations, etc...) does not reflect in any improvement overall.
On the other hand one has to make sure that the available memory bandwidth utilized at best, and that the slow memory accesses are minimized,
since any small improvement there is an overall improvement. 

This is what we try to achieve in this CUDA implementation, making sure data accesses are coalesced whenever possible and caching values in shared memory, while the Dijkstra relaxation operations (floating point arithmetic and control flow) is redundantly computed by all threads.

We use the graphs USA-road-d.NY.gr (264'346 nodes and 733'846 edges), USA-road-d.BAY.gr (321'270 nodes and 800'172 edges), USA-road-d.COL.gr (435'666 nodes and 1'057'066 edges), and compute the shortest paths from 1024 random origins.

![alt text](doc/plot.png)

We see that we get some improvement using our CUDA implementation, not bad considering that GPUs are normally not best suited for graph algorithms in general. We also remark that road networks's connectivity is low when compared to other types of graphs, so the priority queue is not expected to grow very large. It would be interesting to test how does the data structure perform with more highly connected graphs, such as with social media graphs. If you have an NVidia GPU try out the benchmarks yourself and see what you get!

![BibTex citation](citation.bib)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3595244.svg)](https://doi.org/10.5281/zenodo.3595244)
