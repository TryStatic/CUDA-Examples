#include <cstdio>
#include <iostream>
#include "cuda_runtime.h"
#include "device_info.h"
#include <device_launch_parameters.h>
#include "vector_add.h"
#include "util.h"

int main(int argc, char** argv)
{
	if(device_info::get_cuda_capable_devices_count() <= 0)
	{
		printf("No CUDA compute capable device found. Exiting...");
		return -1;
	}

	int exit = 0;
	do
	{
		device_info::print_devices();
		printf("\n\n\n");
		printf("---------------MENU---------------\n");
		printf("\tRun vector addition: 1\n");
		printf("\tExit: 0\n");
		printf("----------------------------------\n\n");

		printf("Enter option: ");
		scanf_s("%d", &exit);

		switch (exit)
		{
			case 0:
				printf("Exiting...\n");
				break;
			case 1:
				printf("Running vector example.\n");
				vector_add::runner();
				break;
			default:
				printf("Invalid selection.\n");
		}

		if (exit != 0)
		{
			printf("\n\nPress enter to continue...\n");
			std::cin.ignore();
			std::cin.get();
			util::clear_screen();
		}
	} while (exit != 0);
	
	return 0;
}
