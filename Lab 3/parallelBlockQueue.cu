#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gputimer.h"

int read_input_one_two_four(int **input1, char* filepath){
 FILE* fp = fopen(filepath, "r");
    if (fp == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    }

    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = ( int *)malloc(len * sizeof(int));

    int temp1;

    while (fscanf(fp, "%d", &temp1) == 1) {
        (*input1)[counter] = temp1;

        counter++;
    }

    fclose(fp);
    return len;

}

int read_input_three(int** input1, int** input2, int** input3, int** input4,char* filepath){
    FILE* fp = fopen(filepath, "r");
    if (fp == NULL){
     fprintf(stderr, "Couldn't open file for reading\n");
     exit(1);
    }

    int counter = 0;
    int len;
    int length = fscanf(fp, "%d", &len);
    *input1 = ( int *)malloc(len * sizeof(int));
    *input2 = ( int *)malloc(len * sizeof(int));
    *input3 = ( int *)malloc(len * sizeof(int));
    *input4 = ( int *)malloc(len * sizeof(int));

    int temp1;
    int temp2;
    int temp3;
    int temp4;
    while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
        (*input1)[counter] = temp1;
        (*input2)[counter] = temp2;
        (*input3)[counter] = temp3;
        (*input4)[counter] = temp4;
        counter++;
    }

    fclose(fp);
    return len;

}

__device__ int gate_solver(int gate, int A, int B) {
    switch (gate) {
        case 0: return A && B;        // AND
        case 1: return A || B;        // OR
        case 2: return !(A && B);     // NAND
        case 3: return !(A || B);     // NOR
        case 4: return A ^ B;         // XOR
        case 5: return !(A ^ B);      // XNOR
        default: return -1;           // Invalid gate type
    }
}

// Global counter for the next level nodes
__device__ int numNextLevelNodes_d = 0; 

// Helper function to add a node to the global queue
__device__ void addToGlobalQueue(int node, int *nextLevelNodes_d) {
    int index = atomicAdd(&numNextLevelNodes_d, 1);
    nextLevelNodes_d[index] = node;
}

__global__ void blockQueuingKernel(int elementsPerThread, int blockNum, int threadNum, int blockQueueCapacity,
                  int *nodePtrs_d, int *nodeGate_d, int numCurrLevelNodes, int *numNextLevelNodes_h,
                  int *currLevelNodes_d, int *nodeNeighbors_d, int *nodeVisited_d,
                  int *nodeInput_d, int *nodeOutput_d, int *nextLevelNodes_d)
{

    // Initialize shared memory for the queue and its size
    extern __shared__ int sharedQueue[];
    __shared__ int sharedQueueSize;

    // Determine the thread ID within its block, the block ID within the grid 
    // and calculate the global index for each thread
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = tid + (bid * blockDim.x);

    // Initialize the shared queue size to 0 by the first thread of the block
    if (tid == 0) sharedQueueSize = 0;
    __syncthreads();

    // Each thread processes multiple elements based on 'elementsPerThread' (Loop over all nodes in the current level)
    for (int idx = i * elementsPerThread; idx < (i+1) * elementsPerThread && idx < numCurrLevelNodes; idx++) {
        
        // Get the current node from the list of current level nodes
        int currNode = currLevelNodes_d[idx];
        
        // Loop over all neighbors of the node
        for (int nbrIdx = nodePtrs_d[currNode]; nbrIdx < nodePtrs_d[currNode + 1]; nbrIdx++) {
            int neighbor = nodeNeighbors_d[nbrIdx];

            // If the neighbor hasn't been visited yet
            if (!nodeVisited_d[neighbor]) {

                // Mark the neighbor as visited
                nodeVisited_d[neighbor] = 1;

                // Compute the output value for the neighbor node
                nodeOutput_d[neighbor] = gate_solver(nodeGate_d[neighbor], nodeOutput_d[currNode], nodeInput_d[neighbor]);

                // Atomically add the neighbor to the shared queue, checking for overflow
                int queueIndex = atomicAdd(&sharedQueueSize, 1);

                // If the queue is not full, add the neighbor to the shared queue
                if (queueIndex < blockQueueCapacity) {
                    sharedQueue[queueIndex] = neighbor;

                // If the queue is full, undo the last increment and add to the global queue
                } else {
                    atomicSub(&sharedQueueSize, 1);
                    addToGlobalQueue(neighbor, nextLevelNodes_d);
                }
            }
        }

        // Synchronize threads to ensure all have processed their nodes
        __syncthreads();

        // Each thread adds its portion of the shared queue to the global queue
        if (tid < sharedQueueSize) {
            addToGlobalQueue(sharedQueue[tid], nextLevelNodes_d);
        }

        // Synchronize threads to ensure global queue update is complete
        __syncthreads();
    }
}

int main(int argc, char *argv[]) {

  // Variables
  int numNodePtrs;
  int numNodes;
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int numTotalNeighbors;
  int *currLevelNodes_h;
  int numCurrLevelNodes;
  int *numNextLevelNodes_h = 0; // Initialize to 0
  int *nodeGate_h;
  int *nodeInput_h;
  int *nodeOutput_h;
  int *nextLevelNodes_h; // Output

  // Parse command line arguments for block queuing
  int blockSize = atoi(argv[1]);
  int numBlocks = atoi(argv[2]);
  int sharedQueueSize = atoi(argv[3]);
  char* input1_path = argv[4];
  char* input2_path = argv[5];
  char* input3_path = argv[6];
  char* input4_path = argv[7];
  char* output_node_path = argv[8];
  char* output_nextnodes_path = argv[9];
  FILE* output_node;
  FILE* output_nextnodes;

  numNodePtrs = read_input_one_two_four(&nodePtrs_h, input1_path);
  numTotalNeighbors = read_input_one_two_four(&nodeNeighbors_h, input2_path);
  numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input3_path);
  numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input4_path);

  nextLevelNodes_h = (int*)malloc(numNodePtrs * sizeof(int));

  // Init Cuda variables
  int *nodePtrs_d;
  int *nodeNeighbors_d;
  int *nodeVisited_d;
  int *currLevelNodes_d;
  int *nodeGate_d;
  int *nodeInput_d;
  int *nodeOutput_d;
  int *nextLevelNodes_d;

  cudaMalloc(&currLevelNodes_d, numCurrLevelNodes * sizeof(int));
  cudaMalloc(&nodePtrs_d, numNodePtrs * sizeof(int));
  cudaMalloc(&nodeNeighbors_d, numTotalNeighbors * sizeof(int));
  cudaMalloc(&nodeVisited_d, numNodes * sizeof(int));
  cudaMalloc(&nodeGate_d, numNodes * sizeof(int));
  cudaMalloc(&nodeInput_d, numNodes * sizeof(int));
  cudaMalloc(&nodeOutput_d, numNodes * sizeof(int));
  cudaMalloc(&nextLevelNodes_d, numNodes * sizeof(int));

  cudaMemcpy(currLevelNodes_d, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodePtrs_d, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h, numTotalNeighbors * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodeGate_d, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodeInput_d, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodeOutput_d, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nextLevelNodes_d, nextLevelNodes_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);

	cudaMallocManaged(&numNextLevelNodes_h, sizeof(int));

  int sharedMemSize = sharedQueueSize * sizeof(int);

  int elementsPerThread = (numCurrLevelNodes + (numBlocks * blockSize) - 1) / (numBlocks * blockSize);

  GpuTimer timer;
  timer.Start();

  blockQueuingKernel <<<numBlocks, blockSize, sharedMemSize>>> (
    elementsPerThread, numBlocks, blockSize, sharedQueueSize, 
    nodePtrs_d, nodeGate_d, numCurrLevelNodes, numNextLevelNodes_h, 
    currLevelNodes_d, nodeNeighbors_d, nodeVisited_d, nodeInput_d, 
    nodeOutput_d, nextLevelNodes_d);
    
  cudaDeviceSynchronize();

  timer.Stop();
  printf("Time Elapsed: %g ms\n", timer.Elapsed());

	//free cuda memory
	cudaMemcpy(nodeOutput_h, nodeOutput_d, (numNodePtrs-1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d, *numNextLevelNodes_h * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(currLevelNodes_d);
	cudaFree(nodePtrs_d);
	cudaFree(nodeNeighbors_d);
	cudaFree(nodeVisited_d);
	cudaFree(nodeGate_d);
	cudaFree(nodeInput_d);
	cudaFree(nodeOutput_d);
	cudaFree(nextLevelNodes_d);

  output_node = fopen(output_node_path, "w");
  if (output_node == NULL) {
      printf("Can't open %s", output_node_path);
      exit(1);
  }

  output_nextnodes = fopen(output_nextnodes_path, "w");
  if (output_nextnodes == NULL) {
      printf("Can't open %s", output_nextnodes_path);
      exit(1);
  }

  fprintf(output_node, "%d\n", numNodePtrs - 1);
  for (int i = 0; i < numNodePtrs - 1; i++) {
      fprintf(output_node, "%d\n", nodeOutput_h[i]);
  }

  fclose(output_node);

  fprintf(output_nextnodes, "%d\n", *numNextLevelNodes_h);
  for (int j = 0; j < *numNextLevelNodes_h; j++) {
      fprintf(output_nextnodes, "%d\n", nextLevelNodes_h[j]);
  }

  fclose(output_nextnodes);
  free(nextLevelNodes_h);

  return 0;
}
