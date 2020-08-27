#include "device.h"

#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>

int device_info::get_cuda_capable_devices_count()
{
	int device_count = 0;
	const cudaError_t error = cudaGetDeviceCount(&device_count);
	if(error != cudaSuccess)
	{
		printf("There was an error while trying to get device count: %s", cudaGetErrorString(error));
		return -1;
	}
	return device_count;
}

void device_info::print_devices()
{
	int device_count;
	cudaGetDeviceCount(&device_count);

	for (int device_id = 0; device_id < device_count; device_id++)
	{
		cudaDeviceProp device_prop{};
		const cudaError_t error = cudaGetDeviceProperties(&device_prop, device_id);
		if (error != cudaSuccess)
		{
			printf("There was an error while trying to get device properties for device id %d: %s", device_id, cudaGetErrorString(error));
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
	printf("\n\nDetected %d device(s) supporting CUDA.\n", device_count);
}
