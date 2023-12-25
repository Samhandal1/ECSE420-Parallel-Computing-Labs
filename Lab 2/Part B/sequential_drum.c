#include <stdio.h>
#include <stdlib.h>

// Constants
const float eta = 0.0002;
const float rho = 0.5;
const float G = 0.75;

// Function to initialize the grid
void initializeGrid(float* u, float* u1, float* u2, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      u[i * N + j] = 0.0f;
      u1[i * N + j] = 0.0f;
      u2[i * N + j] = 0.0f;
    }
  }

  // Set the hit position at (2, 2)
  u1[2 * N + 2] = 1.0f;
}

// Function to update the grid for one iteration
void updateGrid(float* u, float* u1, float* u2, int N) {
  // Inner elements
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      int idx = i * N + j;
      u[idx] = (rho * (u1[(i - 1) * N + j] + u1[(i + 1) * N + j] + u1[i * N + j - 1]
              + u1[i * N + j + 1] - 4.0f * u1[idx])
               + 2.0f * u1[idx] - (1.0f - eta) * u2[idx]) / (1 + eta);
    }
  }

  // Update boundary elements
  for (int i = 1; i < N-1; i++) {
    u[i] = G * u[N + i];
    u[(N - 1) * N + i] = G * u[(N - 2) * N + i];
    u[i * N] = G * u[i * N + 1];
    u[i * N + (N - 1)] = G * u[i * N + (N - 2)];
  }

  // Update corner elements
  u[0] = G * u[N];
  u[(N - 1) * N] = G * u[(N - 2) * N];
  u[N - 1] = G * u[N - 2];
  u[(N - 1) * N + (N - 1)] = G * u[(N - 1) * N + (N - 2)];


  // Set u2 equal to u1 and u1 equal to u
  for (int i = 0; i < N * N; i++) {
      u2[i] = u1[i];
      u1[i] = u[i];
  }
}

void print_u(float *u, int T, int N) {
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

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <number of iterations (T)>\n", argv[0]);
        return 1;
    }

    int T = atoi(argv[1]);
    int N = 4;

    // Allocate memory for the 3 grid iterations
    float* u = (float*) calloc(N * N, sizeof(float));
    float* u1 = (float*) calloc(N * N, sizeof(float));
    float* u2 = (float*) calloc(N * N, sizeof(float));

    initializeGrid(u, u1, u2, N);

    for (int t = 0; t < T; t++) {
        updateGrid(u, u1, u2, N);
        //print_u(u, T, N); // REMOVE BEFORE HANDING IN
        printf("(%d, %d): %f\n", N/2, N/2, u[N/2 * N + N/2]);
    }

    free(u);
    free(u1);
    free(u2);

    return 0;
}
