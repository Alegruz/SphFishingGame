#pragma once

#define WIN32_LEAN_AND_MEAN

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_functions.h> // helper utility functions 
#include <helper_cuda.h>      // helper functions for CUDA error checking and initialization

enum GPU_SELECT_MODE
{
	FIRST_PCI_BUS_ID,
	LAST_PCI_BUS_ID,
	MAX_GFLOPS,
	MIN_GFLOPS,
	SPECIFIED_DEVICE_ID,
	GPU_SELECT_MODE_LAST = SPECIFIED_DEVICE_ID
};

struct GPU_INFO
{
	char	szDeviceName[256];
	DWORD	sm_per_multiproc;
	DWORD	clock_rate;
	DWORD	multiProcessorCount;
	float	TFlops;
};


void ReportGPUMemeoryUsage();
void VerifyCudaError(cudaError err);