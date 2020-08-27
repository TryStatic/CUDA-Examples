#include <cstdio>
#include "cuda_runtime.h"
#include "main.h"
#include "device.h"

int main(int argc, char** argv)
{
	device_info::print_devices();
	return 0;
}
