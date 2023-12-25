#include <stdio.h>
#include <stdlib.h>

// CUDA kernel for parallel logic gate evaluation
__global__ void parallel_logic_gate(int *A, int *B, int *gate, int *output, int N) {

    // Calculate the global index for the current thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure we don't go out of bounds
    if (idx < N) {
      
        // Evaluate the logic gate based on the gate type
        switch (gate[idx]) {
            case 0: output[idx] = A[idx] && B[idx]; break;       // AND
            case 1: output[idx] = A[idx] || B[idx]; break;       // OR
            case 2: output[idx] = !(A[idx] && B[idx]); break;    // NAND
            case 3: output[idx] = !(A[idx] || B[idx]); break;    // NOR
            case 4: output[idx] = A[idx] ^ B[idx]; break;        // XOR
            case 5: output[idx] = !(A[idx] ^ B[idx]); break;     // XNOR
        }
    }
}

int main(int argc, char *argv[]) {

    // Ensure correct number of arguments
    if (argc != 4) {
        printf("Usage: ./parallel_unified <input_file_path> <input_file_length> <output_file_path>\n");
        return 1;
    }

    // Open the input file
    FILE *input_file = fopen(argv[1], "r");
    int N = atoi(argv[2]);
    
    // Declare pointers for inputs and output
    int *A, *B, *gate, *output;

    // Allocate unified memory for inputs and output
    cudaMallocManaged(&A, N*sizeof(int));
    cudaMallocManaged(&B, N*sizeof(int));
    cudaMallocManaged(&gate, N*sizeof(int));
    cudaMallocManaged(&output, N*sizeof(int));

    // Read from the input file into allocated memory
    for (int i = 0; i < N; i++) {
        fscanf(input_file, "%d,%d,%d", &A[i], &B[i], &gate[i]);
    }
    fclose(input_file);

    // Initialize CUDA events to measure data migration time
    cudaEvent_t start, stop;
    float migrationTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Force a migration of Unified Memory to the device by accessing it in a trivial kernel.
    // This might not be the most efficient way, but it allows us to measure the migration time.
    parallel_logic_gate<<<1, 1>>>(A, B, gate, output, 0); // Launching with 1 block and 1 thread just to force migration

    // Wait for GPU tasks to finish to ensure migration is complete
    cudaDeviceSynchronize();

    // Mark the end of data migration to measure time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&migrationTime, start, stop);

    // Initialize CUDA events to measure kernel execution time
    cudaEvent_t kernel_start, kernel_stop;
    float kernelTime;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // Set the kernel launch parameters
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Start recording the kernel execution time
    cudaEventRecord(kernel_start, 0);

    // Launch the CUDA kernel
    parallel_logic_gate<<<blocks, threadsPerBlock>>>(A, B, gate, output, N);

    // Mark the end of kernel execution
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernelTime, kernel_start, kernel_stop);

    // Ensure all GPU tasks are completed before proceeding
    cudaDeviceSynchronize();

    // Open the output file and write results
    FILE *output_file = fopen(argv[3], "w");
    for (int i = 0; i < N; i++) {
        fprintf(output_file, "%d\n", output[i]);
    }
    fclose(output_file);

    // Print out the kernel execution time
    printf("Data migration (host to device) time: %f ms\n", migrationTime);
    printf("CUDA kernel execution time: %f ms\n", kernelTime);

    // Free the allocated unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(gate);
    cudaFree(output);

    // Destroy the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    return 0;
}
