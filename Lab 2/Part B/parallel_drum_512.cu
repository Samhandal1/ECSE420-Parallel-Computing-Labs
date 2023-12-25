#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

// Constants
const float eta = 0.0002;
const float rho = 0.5;
const float G = 0.75;

__global__ void interiorGridKernel(float* u, float* u1, float* u2, int N) {
  int elements_per_thread = (N * N) / (gridDim.x * blockDim.x);
  for (int element = 0; element < elements_per_thread; element++) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx > N * N) return;

    int i = idx / N;  
    int j = idx % N; 

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) { // interior
          u[idx] = (rho * (u1[(i - 1) * N + j] + u1[(i + 1) * N + j] + u1[i * N + j - 1]
                      + u1[i * N + j + 1] - 4.0f * u1[idx])
                      + 2.0f * u1[idx] - (1.0f - eta) * u2[idx]) / (1 + eta);
    }
  }
  return;
}


__global__ void boundaryGridKernel(float* u, float* u1, float* u2, int N) {
  int elements_per_thread = (N * N) / (gridDim.x * blockDim.x);
  for (int element = 0; element < elements_per_thread; element++) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int i = idx / N;  
    int j = idx % N;

    if (i == 0 && j != 0 && j != N - 1) { // row 0, non-corner
        u[idx] = G * u[N + j];
    } else if (i == N - 1 && j != 0 && j != N - 1) { // row N-1, non-corner
        u[idx] = G * u[(N - 2) * N + j];
    } else if (i != 0 && i != N - 1 && j == 0) { // column 0, non-corner
        u[idx] = G * u[i * N + 1];
    } else if (i != 0 && i != N - 1 && j == N - 1) { // column N-1, non-corner
        u[idx] = G * u[i * N + (N - 2)];
    }
  }

  return;
}

__global__ void cornerGridKernel(float* u, float* u1, float* u2, int N) {
  int elements_per_thread = (N * N) / (gridDim.x * blockDim.x);
  for (int element = 0; element < elements_per_thread; element++) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int i = idx / N;  
    int j = idx % N;

    if (i == 0 && j == 0) {
        u[idx] = G * u[N];
    } else if (i == N - 1 && j == 0) {
        u[idx] = G * u[(N - 2) * N];
    } else if (i == 0 && j == N - 1) {
        u[idx] = G * u[N - 2];
    } else if (i == N - 1 && j == N - 1) {
        u[idx] = G * u[(N - 1) * N + (N - 2)];
    }
  }

  return;
}

void print_u(float *u, int N) {
    int row = 0;
    int col = 0;
    for (int i = 0; i < N*N; i++) {
        row = i % N;
        col = i / N;
        printf("(%d, %d): %f\t", row, col, u[i]);
        if (row == N-1) {
          printf("\n");
        }
    }
    printf("\n");
    return;
}

// Function to initialize the grid
void initializeGrid(float* u, float* u1, float* u2, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      u[i * N + j] = 0.0f;
      u1[i * N + j] = 0.0f;
      u2[i * N + j] = 0.0f;
    }
  }

  // Set the hit position at (N/2, N/2)
  u1[(N/2) * N + N/2] = 1.0f;
}


void swap(float **a, float **b) {
    float *temp = *a;
    *a = *b;
    *b = temp;
    return;
}

int drum(int T, int N, int blocks, int threads) 
{
    float grid_size = N*N*sizeof(float);

    float *u, *u1, *u2;
    float *host_u, *host_u1, *host_u2;
    
    // Allocate memory and inialize grid on host
    host_u = (float*)malloc(grid_size);
    host_u1 = (float*)malloc(grid_size);
    host_u2 = (float*)malloc(grid_size);

    initializeGrid(host_u, host_u1, host_u2, N);
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&u, grid_size);
    if (cudaStatus != cudaSuccess) return -1;

    cudaStatus = cudaMalloc((void**)&u1, grid_size);
    if (cudaStatus != cudaSuccess) return -1;
    
    cudaStatus = cudaMalloc((void**)&u2, grid_size);
    if (cudaStatus != cudaSuccess) return -1;
    
    // Copy data from host to device
    cudaMemcpy(u, host_u, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(u1, host_u1, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(u2, host_u2, grid_size, cudaMemcpyHostToDevice);
    
    GpuTimer timer;
    timer.Start();
    for (int t = 0; t < T; t++) {
      
      interiorGridKernel<<<blocks, threads>>> (u, u1, u2, N);
      cudaDeviceSynchronize();
      boundaryGridKernel<<<blocks, threads>>> (u, u1, u2, N);
      cudaDeviceSynchronize();
      cornerGridKernel<<<blocks, threads>>> (u, u1, u2, N);
      cudaDeviceSynchronize();

      cudaMemcpy(host_u, u, grid_size, cudaMemcpyDeviceToHost);        
      //printf("(%d, %d): %f\n", N/2, N/2, host_u[N/2 * N + N/2]);

      swap(&u2, &u1);
      swap(&u1, &u);
    }
    timer.Stop();
    printf("Time Elapsed: %g ms\n", timer.Elapsed());

    cudaFree(u);
    cudaFree(u1);
    cudaFree(u2);
    free(host_u);
    free(host_u1);
    free(host_u2);

    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <number of iterations (T)>\n", argv[0]);
        return 1;
    }

    int T = atoi(argv[1]);
    int N = 512;
    printf("Number of iterations: %d\n", T);
    printf("Grid size: %d\n\n", N*N);

    printf("\n32 blocks, 32 threads, 256 finite element per thread: ");
    drum(T, N, 32, 32);

    printf("\n32 blocks, 128 threads, 64 finite element per thread: ");
    drum(T, N, 32, 128);

    printf("\n32 blocks, 512 threads, 16 finite element per thread: ");
    drum(T, N, 32, 512);

    printf("\n32 blocks, 1024 threads, 8 finite element per thread: ");
    drum(T, N, 32, 1024);

    printf("\n128 blocks, 32 threads, 64 finite element per thread: ");
    drum(T, N, 128, 32);

    printf("\n512 blocks, 32 threads, 16 finite element per thread: ");
    drum(T, N, 512, 32);

    printf("\n1024 blocks, 32 threads, 8 finite element per thread: ");
    drum(T, N, 1024, 32);

    printf("\n128 blocks, 128 threads, 16 finite element per thread: ");
    drum(T, N, 128, 128);

    printf("\n128 blocks, 512 threads, 4 finite element per thread: ");
    drum(T, N, 128, 512);

    printf("\n128 blocks, 1024 threads, 2 finite element per thread: ");
    drum(T, N, 128, 1024);

    printf("\n512 blocks, 128 threads, 4 finite element per thread: ");
    drum(T, N, 512, 128);

    printf("\n1024 blocks, 128 threads, 2 finite element per thread: ");
    drum(T, N, 1024, 128);

    printf("\n512 blocks, 512 threads, 1 finite element per thread: ");
    drum(T, N, 512, 512);

    return 0;
}
