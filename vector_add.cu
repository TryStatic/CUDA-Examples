#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include "cuda_runtime.h"
#include "vector_add.h"

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError(R) {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   return R; \
 }                                                                 \
}

/// <summary>
/// Vector Add Kernel that executes on device
/// </summary>
__global__ void vector_add_kernel(const float *a, const float *b, float  *c, unsigned int N)
{
	// Index calculation
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Safe-check for any extra launched threads
	if(idx < N)
	{
		c[idx] = a[idx] + b[idx];
	}
}

/// <summary>
/// Launches the test case
/// </summary>
int vector_add::runner()
{
	system("CLS");
	printf("\n\n\n[VECTOR_ADD]: STARTING vector_add example.\n");
	
	// -------------------------------------
	// Definitions
	printf("Initizialing definitions\n");

	const unsigned int no_of_elements = 128000000; // amount of total elements in vectors
	const size_t size = no_of_elements * sizeof(float); // required size
	const int threads_per_block = 512; // threads per block
	const int blocks = (int)ceil((float)no_of_elements / threads_per_block); // calculate required blocks

	float ms = 0, total_ms = 0;
	cudaEvent_t kernel_start, kernel_end, memcpy_to_start, memcpy_to_end, memcpy_from_start, memcpy_from_end; // Timing variables
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);
	cudaEventCreate(&memcpy_to_start);
	cudaEventCreate(&memcpy_to_end);
	cudaEventCreate(&memcpy_from_start);
	cudaEventCreate(&memcpy_from_end);

	printf("[Settings] Elements#: %d | reqired size: %d bytes | threads per block: %d | calculated blocks: %d\n\n", no_of_elements, size, threads_per_block, blocks);
	// -------------------------------------


	// -------------------------------------
	// Declare and allocate memory on HOST
	printf("Allocating memory on HOST\n");
	float* h_a = static_cast<float*>(malloc(size));
	float* h_b = static_cast<float*>(malloc(size));
	float* h_c = static_cast<float*>(malloc(size));
	if (h_a == nullptr || h_b == nullptr || h_c == nullptr) return -1;
	printf("DONE\n\n");
	// -------------------------------------


	// -------------------------------------
	// Declare and allocate memory on DEVICE
	printf("Allocating memory on DEVICE\n");
	float* d_a, * d_b, * d_c;
	cudaMalloc(reinterpret_cast<void**>(&d_a), size);
	cudaCheckError(-2);
	cudaMalloc(reinterpret_cast<void**>(&d_b), size);
	cudaCheckError(-2);
	cudaMalloc(reinterpret_cast<void**>(&d_c), size);
	cudaCheckError(-2);
	printf("DONE\n\n");
	// -------------------------------------


	// -------------------------------------
	// Init HOST input vector data
	printf("Initializing HOST input vectors (all set to 1.0f)\n");
	for (int i = 0; i < no_of_elements; i++)
	{
		h_a[i] = 1.0f;
		h_b[i] = 1.0f;
	}
	printf("DONE\n\n");
	// -------------------------------------


	// -------------------------------------
	// Copy HOST Input vectors to device
	printf("COPYING input data from HOST to DEVICE\n");
	cudaEventRecord(memcpy_to_start);
	cudaCheckError(-3);
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaCheckError(-3);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	cudaCheckError(-3);
	cudaEventRecord(memcpy_to_end);
	cudaCheckError(-3);
	printf("DONE\n\n");
	// -------------------------------------


	// -------------------------------------
	// Kernel Launch
	printf("LAUNCHING Kernel\n");
	cudaEventRecord(kernel_start);
	cudaCheckError(-4);
	vector_add_kernel << <blocks, threads_per_block >> > (d_a, d_b, d_c, no_of_elements);
	cudaCheckError(-4);
	printf("WAITING for kernel to finish execution\n");
	cudaDeviceSynchronize(); // BARRIER - Wait for kernel to finish execution
	cudaCheckError(-4);
	cudaEventRecord(kernel_end);
	cudaCheckError(-4);
	printf("KERNEL finished executing\n");
	// -------------------------------------


	// -------------------------------------
	// Copy results back to HOST
	printf("COPYING result data from DEVICE to HOST\n");
	cudaEventRecord(memcpy_from_start);
	cudaCheckError(-5);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	cudaCheckError(-5);
	cudaEventRecord(memcpy_from_end);
	cudaCheckError(-5);
	printf("DONE\n\n");
	// -------------------------------------

	// -------------------------------------
	// Print result
	printf("Device results sample:\n");
	for (int i = 0; i < 10; i++)
	{
		printf("%0.2f ", h_c[i]);
	}
	printf("...\n\n");
	// -------------------------------------


	// -------------------------------------
	// Time events
	cudaEventElapsedTime(&ms, memcpy_to_start, memcpy_to_end);
	cudaCheckError(-6);
	total_ms += ms;
	printf("Memcpy from HOST to DEVICE time: %f sec\n", ms / 1000.0);

	cudaEventElapsedTime(&ms, kernel_start, kernel_end);
	cudaCheckError(-6);
	total_ms += ms;
	printf("KERNEL execution time: %f sec\n", ms / 1000.0);
	
	cudaEventElapsedTime(&ms, memcpy_from_start, memcpy_from_end);
	cudaCheckError(-6);
	total_ms += ms;
	printf("Memcpy from DEVICE to HOST time: %f sec\n\n", ms / 1000.0);
	
	printf("TOTAL Execution time: %f sec\n", total_ms / 1000.0);
	// -------------------------------------

	// -------------------------------------
	// Free memory
	cudaFree(d_a);
	cudaCheckError(-7);
	cudaFree(d_b);
	cudaCheckError(-7);
	cudaFree(d_c);
	cudaCheckError(-7);

	free(h_a);
	free(h_b);
	free(h_c);
	// -------------------------------------

	return 0;
}
