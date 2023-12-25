#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wm.h"

__global__ void convolutionKernel(unsigned char *input, unsigned char *output, float *wm, int width, int height) {
    int output_index = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread index

    if (output_index < (width - 2) * (height - 2)) {
        int j = output_index % (width - 2);
        int i = output_index / (width - 2);

        // For each RGBA value
        for (int k = 0; k <= 3; k++) {
            float sum = 0;
            int integer_sum = 0;
            // Skip Alpha channel
            if (k == 3) {
                integer_sum = 255;
            } else {
                for (unsigned long ii = 0; ii <= 2; ii++) {
                    for (unsigned long jj = 0; jj <= 2; jj++) {
                        sum += 1.0 * input[(i + ii) * width * 4 + (j + jj) * 4 + k] * wm[ii * 3 + jj];
                    }
                }
                integer_sum = round(sum);

                // Ensure value in bounds [0, 255]
                if (integer_sum < 0) {
                    integer_sum = 0;
                } else if (integer_sum > 255) {
                    integer_sum = 255;
                }
            }

            // Write to output
            output[output_index * 4 + k] = integer_sum;
        }
    }
}


int main(int argc, char *argv[])
{
  char* input_filename = argv[1];
  char* output_filename = argv[2];
  int threads = atoi(argv[3]);
  
  unsigned char* input_image;
  unsigned char* output_image;
  unsigned char* host_input_image;
  unsigned char* host_output_image;
  unsigned width, height;
  unsigned error;
  float *wm;

  // Load image from file
  error = lodepng_decode32_file(&host_input_image, &width, &height, input_filename);
  if (error) {
    printf("Error %u: %s\n", error, lodepng_error_text(error));
    return 1;
  }

  // Allocate input and output image size
  int input_image_size = width * height * 4 * sizeof(unsigned char);
  int output_image_size = (width-2) * (height-2) * 4 * sizeof(unsigned char);

  host_output_image = (unsigned char *)malloc(output_image_size);

  cudaMalloc((void **)&input_image, input_image_size);
  cudaMalloc((void **)&output_image, output_image_size);
  cudaMalloc((void **)&wm, 9 * sizeof(float));

  // Copy to host
  cudaMemcpy(input_image, host_input_image, input_image_size, cudaMemcpyHostToDevice);
  cudaMemcpy(wm, w, 9 * sizeof(float), cudaMemcpyHostToDevice);

  // Calculate number of blocks required to complete convolution with <threads> threads
  int blocks = (threads + output_image_size - 1) / threads;

  GpuTimer timer;
  timer.Start();

  convolutionKernel<<<blocks, threads>>>(input_image, output_image, wm, width, height);
  cudaDeviceSynchronize();

  timer.Stop();
  printf("Time Elapsed: %g ms\n", timer.Elapsed());

  cudaMemcpy(host_output_image, output_image, output_image_size, cudaMemcpyDeviceToHost);

  error = lodepng_encode32_file(output_filename, host_output_image, width-2, height-2);
  if (error) {
    printf("Error %u: %s\n", error, lodepng_error_text(error));
    return 1;
  }

  free(host_input_image);
  free(host_output_image);
  cudaFree(input_image);
  cudaFree(output_image);

  return 0;
}
