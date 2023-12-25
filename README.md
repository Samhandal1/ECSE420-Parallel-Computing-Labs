# ECSE420-Parallel-Computing-Labs

In the ECSE 420 labs, the focus was on parallel computing utilizing CUDA. The labs offered practical insights into varied applications, including image processing, logic gate simulation, signal processing, musical instrument synthesis, and the implementation of complex algorithms like breadth-first search.

## Lab 0: Simple CUDA Processing

Objective: Implement and parallelize simple signal processing using CUDA, focusing on image rectification and pooling.

### Tasks:

- Image Rectification: Modify image pixels by centering and rectifying, implementing with CUDA kernels. Performance is measured across various thread counts, requiring a discussion on the parallelization approach and experimental results, including a rectified custom image.
- Pooling: Implement 2x2 max-pooling, compressing the image by selecting maximum values in 2x2 sections. Similar to rectification, analyze and discuss the implementation with an example image.

## Lab 1: Logic Gates Simulation
Objective: Emulate basic logic gates and explore different memory allocation methods in CUDA.

### Tasks:

- Sequential Simulation: Create a sequential code for logic gate simulation.
- Explicit Memory Allocation: Parallelize using explicit memory allocation in CUDA, including performance and data migration time measurements.
- Unified Memory Allocation: Implement the same using unified memory allocation for comparison.
- Optional Advanced Task: Implement unified memory allocation with data prefetching, discussing the specific implementation and its impact.
- Analysis: Discuss the results, focusing on the effectiveness of different memory allocation methods.

## Lab 2: CUDA Convolution and Musical Instrument Simulation

Objective: Develop parallel computing solutions for signal processing through image convolution and simulate musical instrument sounds using the finite element method.

### Tasks:

A. Convolution:
- Implement a 3x3 convolution operation for image processing, ensuring output value constraints.
- Test with provided images, analyze runtime performance across various thread counts, and discuss the parallelization strategy.

B. Finite Element Music Synthesis:
- Implement a drum sound simulation using a 2D grid of finite elements.
- Develop sequential and parallelized versions, experimenting with different thread-block-element combinations.
- Provide execution time comparisons and a thorough analysis of the parallelization approaches.

## Lab 3: Breadth-First Search (BFS) with CUDA

Objective: Implement a single iteration of BFS using shared memory and atomic operations, and parallelize it using CUDA.

### Tasks:

- Sequential BFS: Develop a sequential BFS algorithm.
- Global Queuing: Parallelize using global queuing with varying blockSize and numBlock.
- Block Queuing: Implement block queuing using shared memory, with different configurations of blockSize, numBlock, and blockQueueCapacity.
- Analysis: Discuss the implementations, results, and the architecture used in experiments.
