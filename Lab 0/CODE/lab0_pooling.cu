#include "lodepng.h"
#include "gputimer.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void poolingKernel(unsigned char* input_image, unsigned char* output_image, int width)
{
  // Calculate the global index of the current thread across all blocks
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Determine the x, y position of the pixel in the input image
  // Since the output image is half the size of the input image,
  // we need to multiply by 2 to get the corresponding position in the input
  int x = (idx % (width/2)) * 2;
  int y = (idx / (width/2)) * 2;

  // Calculate the starting index of the 2x2 block in the input image, times 4 for RGBA channels
  int input_index = (y * width + x) * 4;

  // Calculate the corresponding index in the output image, times 4 for RGBA channels
  int output_index = idx * 4;

  // Iterate over all RGBA channels
  for(int channel = 0; channel < 4; channel++) {

      // To store the maximum value in the 2x2 block
      unsigned char max_val = 0;

      // Iterate over the 2x2 block for the given channel
      for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {

              // Calculate the current pixel index within the 2x2 block
              int cur_index = (input_index + i * width * 4 + j * 4) + channel;

              // Update max_val with the maximum value in the 2x2 block for the current channel
              max_val = max(max_val, input_image[cur_index]);
          }
      }

      // Store the maximum value in the output image for the current channel
      output_image[output_index + channel] = max_val;
  }

}

int pooling(char* input_filename, char* output_filename, int num_threads)
{
  unsigned error;
  unsigned char* image;
  unsigned char* tmp_image;
  unsigned char* pool_image;
  unsigned char* output_image;
  unsigned width;
  unsigned height;

  // Load image into array
  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  // Determine memory size to allocate for image
  unsigned int image_size_bytes = width * height * 4 * sizeof(unsigned char);
  unsigned int pool_image_size_bytes = (width/2) * (height/2) * 4 * sizeof(unsigned char);

  // Allocate device memory for image storage & pooled image storage
  cudaError_t cudaStatus = cudaMalloc(&tmp_image, image_size_bytes);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));

  cudaStatus = cudaMalloc(&pool_image, pool_image_size_bytes);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));

  // Copy the image data to device
  cudaStatus = cudaMemcpy(tmp_image, image, image_size_bytes, cudaMemcpyHostToDevice);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));


  // Determine how many blocks are required of num_threads threads to cover all bytes
  int total_pixels_to_process = (width/2) * (height/2);
  int num_blocks = (total_pixels_to_process + num_threads - 1) / num_threads;


  // Initialize and run timer to calculate execution time
  struct GpuTimer timer;
  timer.Start();

  // Launch kernel on image
  poolingKernel<<<num_blocks,num_threads>>>(tmp_image, pool_image, width);
  cudaDeviceSynchronize();
  timer.Stop();
  printf("Runtime of %f ms for %d threads\n", timer.Elapsed(), num_threads);

  // Copy the image from device back to host
  output_image = (unsigned char*) malloc(pool_image_size_bytes);
  cudaStatus = cudaMemcpy(output_image, pool_image, pool_image_size_bytes, cudaMemcpyDeviceToHost);
  if(cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));

  // Write the rectified image to output file
  lodepng_encode32_file(output_filename, output_image, (width/2), (height/2));

  cudaFree(tmp_image);
  cudaFree(pool_image);
  free(image);
  free(output_image);
  return 0;
}

int main(int argc, char *argv[])
{
  if(argc != 4)
  {
    printf("Usage error: ./pool <name of input png> <name of output png> <# threads>.");
    return -1;
  }

  char* input_filename = argv[1];
  char* output_filename = argv[2];
  int num_threads = atoi(argv[3]);

  int poolingError = pooling(input_filename, output_filename, num_threads);
  return poolingError;
}
