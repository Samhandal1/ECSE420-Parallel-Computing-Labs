#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "read_input.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "read_input.h"

int gate_solver(int gate, int A, int B) {
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

int main(int argc, char *argv[]) {
    // Variables
    int numNodePtrs;
    int numNodes;
    int *nodePtrs_h;
    int *nodeNeighbors_h;
    int *nodeVisited_h;
    int numTotalNeighbors_h;
    int *currLevelNodes_h;
    int numCurrLevelNodes;
    int numNextLevelNodes_h = 0; // Initialize to 0
    int *nodeGate_h;
    int *nodeInput_h;
    int *nodeOutput_h;

    // Output
    int *nextLevelNodes_h;

    char* input1_path = argv[1];
    char* input2_path = argv[2];
    char* input3_path = argv[3];
    char* input4_path = argv[4];

    char* output_node_path = argv[5];
    char* output_nextnodes_path = argv[6];
    FILE* output_node;
    FILE* output_nextnodes;

    numNodePtrs = read_input_one_two_four(&nodePtrs_h, input1_path);
    numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, "input2.raw");
    numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, "input3.raw");
    numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, "input4.raw");

    nextLevelNodes_h = (int*)malloc(numNodePtrs * sizeof(int));

    clock_t begin = clock();

    // Loop over all nodes in the current level
    for (int idx = 0; idx < numCurrLevelNodes; idx++) {
        int currNode = currLevelNodes_h[idx];
        // Loop over all neighbors of the node
        for (int nbrIdx = nodePtrs_h[currNode]; nbrIdx < nodePtrs_h[currNode + 1]; nbrIdx++) {
            int neighbor = nodeNeighbors_h[nbrIdx];
            // If the neighbor hasn't been visited yet
            if (!nodeVisited_h[neighbor]) {
                // Mark it and add it to the queue
                nodeVisited_h[neighbor] = 1;
                nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[currNode], nodeInput_h[neighbor]);
                nextLevelNodes_h[numNextLevelNodes_h] = neighbor;
                ++numNextLevelNodes_h;
            }
            nodeVisited_h[currNode] = 1;
        }
    }

    clock_t end = clock();

    float elapsed_time = ((double)end - begin) / CLOCKS_PER_SEC * 1000;
    printf("Elapsed time is %f ms\n", elapsed_time);

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

    fprintf(output_nextnodes, "%d\n", numNextLevelNodes_h);
    for (int j = 0; j < numNextLevelNodes_h; j++) {
        fprintf(output_nextnodes, "%d\n", nextLevelNodes_h[j]);
    }

    fclose(output_nextnodes);
    free(nextLevelNodes_h);

    return 0;
}
