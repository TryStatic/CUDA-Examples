#include <iostream>
#include "cuda_runtime.h"
#include "main.h"


int main(int argc, char** argv)
{
	print_cuda_devices();
	return 0;
}

void print_cuda_devices()
{

	int device_count;
	cudaGetDeviceCount(&device_count);

	for (int device_id = 0; device_id < device_count; device_id++)
	{
		cudaDeviceProp device_prop;
		const cudaError_t error = cudaGetDeviceProperties(&device_prop, device_id);
		if(error != cudaSuccess)
		{
			printf("There was an error while trying to get device properties for device %d", device_id);
			return;
		}
		
		if (device_id == 0)
		{
			if (device_prop.major == 9999 && device_prop.minor == 9999)
			{
				printf("No CUDA GPU has been detected.\n");
				return;
			}

			printf("Device %d: %s\n", device_id, device_prop.name);
			printf("\tComputational Capabilities: %d.%d\n", device_prop.major, device_prop.minor);
			printf("\tMaximum global memory size: %llu\n", device_prop.totalGlobalMem);
			printf("\tMaximum constant memory size: %llu\n", device_prop.totalConstMem);
			printf("\tMaximum shared memory size per block: %llu\n", device_prop.sharedMemPerBlock);
			printf("\tMaximum block dimensions: %d x %d x %d\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
			printf("\tMaximum grid dimensions: %d x %d x %d\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
			printf("\tWarp size: %d\n", device_prop.warpSize);
		}
	}

	std::cout << std::endl << std::endl;

	printf("Detected %d device(s) supporting CUDA.\n", device_count);
}
