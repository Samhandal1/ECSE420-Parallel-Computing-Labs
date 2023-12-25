#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void rectifyKernel(unsigned char* image)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (image[i] < 127) {
    image[i] = 127;
  }
}

int rectify(char* input_filename, char* output_filename, int num_threads)
{
  unsigned error;
  unsigned char* image;
  unsigned char* tmp_image;
  unsigned width;
  unsigned height;
  unsigned int image_size_bytes;

  // Load image into array
  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  // Determine memory size to allocate for image
  image_size_bytes = width * height * 4 * sizeof(unsigned char);

  // Allocate device memory for image storage
  cudaError_t cudaStatus = cudaMalloc(&tmp_image, image_size_bytes);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));

  // Copy the image data to device
  cudaStatus = cudaMemcpy(tmp_image, image, image_size_bytes, cudaMemcpyHostToDevice);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));


  // Determine how many blocks are required of num_threads threads to cover all bytes
  int num_blocks = (width * height * 4 + num_threads - 1) / num_threads;

  // Initialize and run timer to calculate execution time
  struct GpuTimer timer;
  timer.Start();

  // Launch kernel on image
  rectifyKernel<<<num_blocks,num_threads>>>(tmp_image);
  cudaDeviceSynchronize();
  timer.Stop();
  printf("Runtime of %f ms for %d threads\n", timer.Elapsed(), num_threads);

  // Copy the image from device back to host
  cudaStatus = cudaMemcpy(image, tmp_image, image_size_bytes, cudaMemcpyDeviceToHost);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));

  // Write the rectified image to output file
  lodepng_encode32_file(output_filename, image, width, height);

  cudaFree(tmp_image);
  free(image);
  return 0;
}

int main(int argc, char *argv[])
{
  if(argc != 4)
  {
    printf("Error: requires 3 inputs.");
    return -1;
  }
  char* input_filename = argv[1];
  char* output_filename = argv[2];
  int num_threads = atoi(argv[3]);

  int rectifyError = rectify(input_filename, output_filename, num_threads);
  return rectifyError;
}
