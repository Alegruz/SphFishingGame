#include "CUDA_Util.h"

void ReportGPUMemeoryUsage()
{
	size_t free_byte = 0;
	size_t total_byte = 0;
	cudaError cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
}
void VerifyCudaError(cudaError err)
{
	WCHAR*	wchErr = L"Unknown";
	switch (err)
	{
		case cudaErrorMemoryAllocation:
			wchErr = L"cudaErrorMemoryAllocation";
			break;

		case cudaErrorLaunchFailure:
			wchErr = L"cudaErrorLaunchFailure";
			break;
		case cudaErrorLaunchTimeout:
			wchErr = L"cudaErrorLaunchTimeout";
			break;

		case cudaErrorMisalignedAddress:
			wchErr = L"cudaErrorMisalignedAddress";
			break;
		case cudaErrorInvalidValue:
			wchErr = L"cudaErrorInvalidValue";
			break;

	}
	if (err != cudaSuccess)
	{
		__debugbreak();
	}
}