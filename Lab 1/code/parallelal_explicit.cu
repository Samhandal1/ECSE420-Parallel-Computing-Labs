#include <stdio.h>
#include <stdlib.h>

// CUDA kernel function to perform logic gate operations on input arrays
__global__ void parallel_logic_gate(int *d_A, int *d_B, int *d_gate, int *d_output, int N) {
    // Calculate the unique index for the current thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread is within bounds
    if (idx < N) {
        // Evaluate logic based on gate value
        switch (d_gate[idx]) {
            case 0: d_output[idx] = d_A[idx] && d_B[idx]; break;      // AND
            case 1: d_output[idx] = d_A[idx] || d_B[idx]; break;      // OR
            case 2: d_output[idx] = !(d_A[idx] && d_B[idx]); break;   // NAND
            case 3: d_output[idx] = !(d_A[idx] || d_B[idx]); break;   // NOR
            case 4: d_output[idx] = d_A[idx] ^ d_B[idx]; break;       // XOR
            case 5: d_output[idx] = !(d_A[idx] ^ d_B[idx]); break;    // XNOR
        }
    }
}

int main(int argc, char *argv[]) {

    // Check if correct number of command line arguments are provided
    if (argc != 4) {
        printf("Usage: ./parallel_explicit <input_file_path> <input_file_length> <output_file_path>\n");
        return 1;
    }

    // Open the input file for reading
    FILE *input_file = fopen(argv[1], "r");
    int N = atoi(argv[2]); // Number of logic operations to perform

    // Pointers for host and device memory
    int *h_A, *h_B, *h_gate, *h_output;
    int *d_A, *d_B, *d_gate, *d_output;

    // Allocate memory on the host for input data and results
    h_A = (int*)malloc(N * sizeof(int));
    h_B = (int*)malloc(N * sizeof(int));
    h_gate = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc(N * sizeof(int));

    // Read input data from the file
    for (int i = 0; i < N; i++) {
        fscanf(input_file, "%d,%d,%d", &h_A[i], &h_B[i], &h_gate[i]);
    }
    fclose(input_file);

    // Allocate memory on the GPU for input data and results
    cudaMalloc(&d_A, N*sizeof(int));
    cudaMalloc(&d_B, N*sizeof(int));
    cudaMalloc(&d_gate, N*sizeof(int));
    cudaMalloc(&d_output, N*sizeof(int));

    // Initialize CUDA events to measure data migration time
    cudaEvent_t start, stop;
    float migrationTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Transfer input data from host to GPU
    cudaMemcpy(d_A, h_A, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_gate, N*sizeof(int), cudaMemcpyHostToDevice);

    // Mark the end of data transfer and calculate the time taken
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&migrationTime, start, stop);

    // Initialize CUDA events to measure kernel execution time
    cudaEvent_t kernel_start, kernel_stop;
    float kernelTime;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // Define the grid and block sizes for CUDA kernel execution
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Start recording the kernel execution time
    cudaEventRecord(kernel_start, 0);

    // Launch the CUDA kernel
    parallel_logic_gate<<<blocks, threadsPerBlock>>>(d_A, d_B, d_gate, d_output, N);

    // Mark the end of kernel execution
    cudaEventRecord(kernel_stop, 0);
    cudaEventSynchronize(kernel_stop);
    cudaEventElapsedTime(&kernelTime, kernel_start, kernel_stop);

    // Ensure all GPU operations are complete
    cudaDeviceSynchronize();

    // Transfer the computed results back to the host
    cudaMemcpy(h_output, d_output, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Write the results to the output file
    FILE *output_file = fopen(argv[3], "w");
    for (int i = 0; i < N; i++) {
        fprintf(output_file, "%d\n", h_output[i]);
    }
    fclose(output_file);

    // Print out the data migration time and kernel execution time
    printf("Data migration (host to device) time: %f ms\n", migrationTime);
    printf("CUDA kernel execution time: %f ms\n", kernelTime);

    // Free all the allocated memory
    free(h_A);
    free(h_B);
    free(h_gate);
    free(h_output);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_gate);
    cudaFree(d_output);

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    return 0;
}
