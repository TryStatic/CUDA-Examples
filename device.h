#pragma once

struct device_info
{
	static int get_cuda_capable_devices_count();
	static void print_devices();
};
