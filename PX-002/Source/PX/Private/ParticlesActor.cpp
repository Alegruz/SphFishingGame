// Fill out your copyright notice in the Description page of Project Settings.


#include "ParticlesActor.h"
#include "Components/InstancedStaticMeshComponent.h"

#include "defines.h"

#define DEBUG_BUFFERS 0

// Sets default values
AParticlesActor::AParticlesActor()
	: SupportRadius(ParticleRadius * 2.0f)
	, SupportRadiusSquared(SupportRadius* SupportRadius)
	, ThresholdSquared(Threshold * Threshold)
	, Poly6(315.0f / (64.0f * PI * FMath::Pow(SupportRadius, 9.0f)))
	, Poly6Gradient(-(945.0f / (32.0f * PI * pow(SupportRadius, 9.0f))))
	, Poly6Laplacian(-(945.0f / (32.0f * PI * pow(SupportRadius, 9.0f))))
	, SpikyGradient(-(45.0f / (PI * pow(SupportRadiusSquared, 3.0f))))
	, ViscosityLaplacian(45.0f / (PI * pow(SupportRadiusSquared, 3.0f)))
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	ParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>("ParticleInstancedMeshComponent");
	ParticleProceduralMeshComponent = CreateDefaultSubobject<UProceduralMeshComponent>("ParticleProceduralMeshComponent");
//#if RENDER_INSTANCES
//	RootComponent = ParticleInstancedMeshComponent;
//#else
//	RootComponent = ParticleProceduralMeshComponent;
//#endif
	ParticleProceduralMeshComponent->bUseAsyncCooking = true;

	CudaSelectedDeviceId = -1;
	GpuInfo = {};
	bCanPrefetch = false;

	if (SphPlatform == ESphPlatform::E_GPU_CUDA)
	{
		InitializeCuda(SPECIFIED_DEVICE_ID, 0);
		//char* Argv[] = { "Sph" };
		//cudaInit(1u, Argv);
	}


	GridSize = make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE);
	SolverIterations = 1u;

	NumGridCells = GridSize.x * GridSize.y * GridSize.z;
	GridSortBits = 18u;	// increase this for larger grids

	// set simulation parameters
	SimulationParameters.GridSize = GridSize;
	SimulationParameters.NumCells = NumGridCells;
	SimulationParameters.NumBodies = NumParticles;

	SimulationParameters.ParticleRadius = ParticleRadius;

	SimulationParameters.WorldOrigin = make_float3(-1.0f, -1.0f, -1.0f);

	SimulationParameters.BoundaryDamping = BoundaryDamping;

	SimulationParameters.ParticleMass = Mass;
	SimulationParameters.SupportRadius = SupportRadius;
	SimulationParameters.SupportRadiusSquared = SupportRadiusSquared;

	const double CellSize = SimulationParameters.SupportRadius;  // cell size equal to particle diameter
	SimulationParameters.CellSize = make_float3(CellSize, CellSize, CellSize);

	SimulationParameters.GasConstant = StiffnessConstant;
	SimulationParameters.RestDensity = RestDensity;
	SimulationParameters.Viscosity = Viscosity;
	SimulationParameters.Threshold = Threshold;
	SimulationParameters.ThresholdSquared = ThresholdSquared;
	SimulationParameters.SurfaceTension = SurfaceTension;

	SimulationParameters.CflScale = CflScale;

	SimulationParameters.Poly6 = Poly6;
	SimulationParameters.Poly6Gradient = Poly6Gradient;
	SimulationParameters.Poly6Laplacian = Poly6Laplacian;
	SimulationParameters.SpikyGradient = SpikyGradient;
	SimulationParameters.ViscosityLaplacian = ViscosityLaplacian;

	SimulationParameters.Gravity = make_float3(Gravity.X, Gravity.Z, Gravity.Y);
	SimulationParameters.DeltaTime = CustomDeltaTime;

	//SimulationParameters.XScaleFactor = GetActorScale().X;
	//SimulationParameters.YScaleFactor = GetActorScale().Z;
	//SimulationParameters.ZScaleFactor = GetActorScale().Y;
	SimulationParameters.XScaleFactor = 0.5f;
	SimulationParameters.YScaleFactor = 1.0f;
	SimulationParameters.ZScaleFactor = 1.0f;

	// Initialize CUDA buffers for Marching Cubes
	GridSizeLog2 = make_uint3(GRID_SIZE_LOG2, GRID_SIZE_LOG2, GRID_SIZE_LOG2);
	NumVoxels = 0;
	NumMaxVertice = 0;
	NumActiveVoxels = 0;
	NumTotalVertice = 0;

	//IsoValue = 0.2f;
	//DeviceIsoValue = 0.005f;
	IsoValue = 0.0f;
	DeviceIsoValue = 0.0f;

	HostMcVertices = nullptr;
	HostMcNormals = nullptr;

	// device data
	DeviceMcPositions = nullptr;
	DeviceMcNormals = nullptr;

	DeviceVolumes = nullptr;
	DeviceVoxelVertices = nullptr;
	DeviceVoxelVerticesScan = nullptr;
	DeviceVoxelsOccupied = nullptr;
	DeviceVoxelsOccupiedScan = nullptr;
	DeviceCompactedVoxelArray;

	// tables
	DeviceNumVerticesTable = nullptr;
	DeviceEdgeTable = nullptr;
	DeviceTriTable = nullptr;

	//const int ArraySize = 5;
	//const int A[ArraySize] = { 1, 2, 3, 4, 5 };
	//const int B[ArraySize] = { 10, 20, 30, 40, 50 };
	//int C[ArraySize] = { 0 };
	//int ErrorCode = -1;

	//// Add vectors in parallel
	//cudaError_t cudaStatus = addWithCuda(C, A, B, ArraySize, &ErrorCode);
	//if (cudaStatus != cudaSuccess)
	//{
	//	UE_LOG(LogTemp, Warning, TEXT("addWithCuda failed!"));
	//	UE_LOG(LogTemp, Warning, TEXT("%d, %d"), cudaStatus, ErrorCode);
	//	return;
	//}
	//UE_LOG(LogTemp, Warning, TEXT("{1, 2, 3, 4, 5} + {10, 20, 30, 40, 50} = {%d, %d, %d, %d, %d}"), C[0], C[1], C[2], C[3], C[4]);

	UE_LOG(LogTemp, Warning, TEXT("PARAMETERS"));
	UE_LOG(LogTemp, Warning, TEXT("PARTICLE RADIUS=%f"), SimulationParameters.ParticleRadius);
	UE_LOG(LogTemp, Warning, TEXT("SUPPORT RADIUS=%f"), SimulationParameters.SupportRadius);
	UE_LOG(LogTemp, Warning, TEXT("SUPPORT RADIUS SQUARED=%f"), SimulationParameters.SupportRadiusSquared);
	UE_LOG(LogTemp, Warning, TEXT("STIFFNESS CONSTANT=%f"), SimulationParameters.GasConstant);
	UE_LOG(LogTemp, Warning, TEXT("REST DENSITY=%f"), SimulationParameters.RestDensity);
	UE_LOG(LogTemp, Warning, TEXT("VISCOSITY=%f"), SimulationParameters.Viscosity);
	UE_LOG(LogTemp, Warning, TEXT("GRAVITY=%f, %f, %f"), SimulationParameters.Gravity.x, SimulationParameters.Gravity.y, SimulationParameters.Gravity.z);
	UE_LOG(LogTemp, Warning, TEXT("POLY6=%f"), SimulationParameters.Poly6);
	UE_LOG(LogTemp, Warning, TEXT("POLY6 GRADIENT=%f"), SimulationParameters.Poly6Gradient);
	UE_LOG(LogTemp, Warning, TEXT("POLY6 LAPLACIAN=%f"), SimulationParameters.Poly6Laplacian);
	UE_LOG(LogTemp, Warning, TEXT("SPIKY GRADIENT=%f"), SimulationParameters.SpikyGradient);
	UE_LOG(LogTemp, Warning, TEXT("VISCOSITY LAPLACIAN=%f"), SimulationParameters.ViscosityLaplacian);
	UE_LOG(LogTemp, Warning, TEXT("SCALE=%f, %f, %f"), SimulationParameters.XScaleFactor, SimulationParameters.YScaleFactor, SimulationParameters.ZScaleFactor);
	UE_LOG(LogTemp, Warning, TEXT("GRID SIZE=%u, %u, %u"), SimulationParameters.GridSize.x, SimulationParameters.GridSize.y, SimulationParameters.GridSize.z);
	switch (SphPlatform)
	{
	case ESphPlatform::E_CPU_SINGLE_THREAD:
		UE_LOG(LogTemp, Warning, TEXT("PLATFORM=CPU Single Thread"));
		break;
	case ESphPlatform::E_CPU_MULTIPLE_THREADS:
		UE_LOG(LogTemp, Warning, TEXT("PLATFORM=CPU Multiple Threads"));
		break;
	case ESphPlatform::E_GPU_CUDA:
		UE_LOG(LogTemp, Warning, TEXT("PLATFORM=GPU CUDA"));
		break;
	default:
		break;
	}

	//Initialize();
}

AParticlesActor::~AParticlesActor()
{
	Destroy();
	NumParticles = 0;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		UE_LOG(LogTemp, Error, TEXT("cudaDeviceReset failed!"));
	}
}

// Called when the game starts or when spawned
void AParticlesActor::BeginPlay()
{
	Super::BeginPlay();

	Initialize();
	Reset();
}

void AParticlesActor::Initialize()
{
	uint32 MemorySize = sizeof(float) * 4u * NumParticles;

	// allocate host storage
	HostPositions = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostVelocities = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostPressureForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostViscosityForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	//HostSurfaceTensionForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostDensities = reinterpret_cast<float*>(FMemory::Malloc(sizeof(float) * NumParticles));
	HostPressures = reinterpret_cast<float*>(FMemory::Malloc(sizeof(float) * NumParticles));

	FMemory::Memzero(reinterpret_cast<void*>(HostPositions), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostVelocities), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostForces), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostPressureForces), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostViscosityForces), MemorySize);
	//FMemory::Memzero(reinterpret_cast<void*>(HostSurfaceTensionForces), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostDensities), sizeof(float) * NumParticles);
	FMemory::Memzero(reinterpret_cast<void*>(HostPressures), sizeof(float) * NumParticles);

	HostCellStarts = reinterpret_cast<uint32*>(FMemory::Malloc(sizeof(uint32) * NumGridCells));
	FMemory::Memzero(reinterpret_cast<void*>(HostCellStarts), sizeof(uint32) * NumGridCells);

	HostCellEnds = reinterpret_cast<uint32*>(FMemory::Malloc(sizeof(uint32) * NumGridCells));
	FMemory::Memzero(reinterpret_cast<void*>(HostCellEnds), sizeof(uint32) * NumGridCells);

	if (SphPlatform == ESphPlatform::E_GPU_CUDA)
	{
		// allocate GPU data
		cudaMalloc(reinterpret_cast<void**>(&CudaPositionsVbo), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceVelocities), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceForces), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DevicePressureForces), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceViscosityForces), MemorySize);
		//cudaMalloc(reinterpret_cast<void**>(&DeviceSurfaceTensionForces), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceDensities), sizeof(float) * NumParticles);
		cudaMalloc(reinterpret_cast<void**>(&DevicePressures), sizeof(float) * NumParticles);

		cudaMalloc(reinterpret_cast<void**>(&DeviceSortedPositions), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceSortedVelocities), MemorySize);

		cudaMalloc(reinterpret_cast<void**>(&DeviceGridParticleHashes), sizeof(uint32) * NumParticles);
		cudaMalloc(reinterpret_cast<void**>(&DeviceGridParticleIndice), sizeof(uint32) * NumParticles);

		cudaMalloc(reinterpret_cast<void**>(&DeviceCellStarts), sizeof(uint32) * NumGridCells);
		cudaMalloc(reinterpret_cast<void**>(&DeviceCellEnds), sizeof(uint32) * NumGridCells);

		CudaSetParameters(&SimulationParameters);
	}

#if !RENDER_INSTANCES
	GridSizeMask = make_uint3(GridSize.x - 1, GridSize.y - 1, GridSize.z - 1);
	GridSizeShift = make_uint3(0, GridSizeLog2.x, GridSizeLog2.x + GridSizeLog2.y);

	NumVoxels = GridSize.x * GridSize.y * GridSize.z;
	VoxelSize = make_float3(2.0f / GridSize.x, 2.0f / GridSize.y, 2.0f / GridSize.z);
	NumMaxVertice = GridSize.x * GridSize.y * 100;

	printf("Grid: %d x %d x %d = %d voxels\n", GridSize.x, GridSize.y, GridSize.z, NumVoxels);
	printf("max verts = %d\n", NumMaxVertice);

	cudaMalloc((void**)&(DeviceMcPositions), NumMaxVertice * sizeof(float) * 4);
	cudaMalloc((void**)&(DeviceMcNormals), NumMaxVertice * sizeof(float) * 4);
	cudaMallocHost(reinterpret_cast<void**>(&HostMcVertices), NumMaxVertice * sizeof(float) * 4);
	cudaMallocHost(reinterpret_cast<void**>(&HostMcNormals), NumMaxVertice * sizeof(float) * 4);

	// allocate textures
	CudaAllocateTextures(&DeviceEdgeTable, &DeviceTriTable, &DeviceNumVerticesTable);

	// allocate device memory
	MemorySize = sizeof(uint32) * NumVoxels;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelVertices), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelVerticesScan), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelsOccupied), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelsOccupiedScan), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceCompactedVoxelArray), MemorySize));

	Vertices.Reserve(NumMaxVertice);
	Triangles.Reserve(NumMaxVertice);
	Uv0.Reserve(NumMaxVertice);
#else
	ParticleInstancedMeshComponent->PreAllocateInstancesMemory(NumParticles);
	UE_LOG(LogTemp, Warning, TEXT("Initialize with %u Particles"), NumParticles);
	for (uint32 i = 0; i < NumParticles; ++i)
	{
		FVector Location(0.0f, 0.0f, 0.0f);
		int32 Result = ParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(ParticleRenderRadius / ParticleMeshRadius)));
		UE_LOG(LogTemp, Warning, TEXT("Adding Instance... %d / %d"), ParticleInstancedMeshComponent->GetInstanceCount(), Result);
	}
	UE_LOG(LogTemp, Warning, TEXT("Initialize::Num Render Instances: %d"), ParticleInstancedMeshComponent->GetNumRenderInstances());
#endif
}

bool AParticlesActor::InitializeCuda()
{
	return InitializeCuda(SPECIFIED_DEVICE_ID, 0);
}

bool AParticlesActor::InitializeCuda(GPU_SELECT_MODE Mode, int32 SpecifiedDeviceId)
{
	bool bResult = false;

	int32 DeviceCount = 0;

	if (cudaSuccess != cudaGetDeviceCount(&DeviceCount))
	{
		goto lb_return;
	}

	struct CudaDeviceProperty
	{
		cudaDeviceProp Prop;
		int32 DeviceId;
	};

	enum eCudaDeviceType
	{
		CUDA_DEVICE_FIRST_PCI_BUS_ID,
		CUDA_DEVICE_LAST_PCI_BUS_ID,
		CUDA_DEVICE_MAX_GFLOPS,
		CUDA_DEVICE_MIN_GFLOPS,
		CUDA_DEVICE_SPECIFIED
	};
	const uint32 CUDA_DEVICE_TYPE_NUM = 5;
	CudaDeviceProperty DeviceProp[CUDA_DEVICE_TYPE_NUM] = {};

	for (uint32 i = 0; i < CUDA_DEVICE_TYPE_NUM; ++i)
	{
		DeviceProp[i].DeviceId = -1;
	}

	if (DeviceCount == 0)
	{
		goto lb_return;
	}

	int32 LastPciBusId = -1;
	int32 FirstPciBusId = INT32_MAX;
	int32 MaxGFlops = -1;
	int32 MinGFlops = INT32_MAX;

	cudaDeviceProp Prop;
	for (int32 i = 0; i < DeviceCount; ++i)
	{
		if (cudaSuccess != cudaGetDeviceProperties(&Prop, i))
		{
			__debugbreak();
		}

		if (Prop.major < 2)
		{
			continue;
		}

		if (i == SpecifiedDeviceId)
		{
			DeviceProp[CUDA_DEVICE_SPECIFIED].Prop = Prop;
			DeviceProp[CUDA_DEVICE_SPECIFIED].DeviceId = i;
		}

		if (Prop.pciBusID > LastPciBusId)
		{
			LastPciBusId = Prop.pciBusID;
			DeviceProp[CUDA_DEVICE_LAST_PCI_BUS_ID].Prop = Prop;
			DeviceProp[CUDA_DEVICE_LAST_PCI_BUS_ID].DeviceId = i;
		}

		if (Prop.pciBusID > FirstPciBusId)
		{
			FirstPciBusId = Prop.pciBusID;
			DeviceProp[CUDA_DEVICE_FIRST_PCI_BUS_ID].Prop = Prop;
			DeviceProp[CUDA_DEVICE_FIRST_PCI_BUS_ID].DeviceId = i;
		}

		float ClockRate = static_cast<float>(Prop.clockRate / (1000.0f * 1000.0f));
		int32 SmPerMultiproc = _ConvertSMVer2Cores(Prop.major, Prop.minor);
		float GFlops = static_cast<float>(Prop.multiProcessorCount * SmPerMultiproc) * ClockRate * 2.0f;

		if (GFlops > MaxGFlops)
		{
			MaxGFlops = static_cast<int32>(GFlops);
			DeviceProp[CUDA_DEVICE_MAX_GFLOPS].Prop = Prop;
			DeviceProp[CUDA_DEVICE_MAX_GFLOPS].DeviceId = i;
		}

		if (GFlops < MinGFlops)
		{
			MinGFlops = static_cast<int32>(GFlops);
			DeviceProp[CUDA_DEVICE_MIN_GFLOPS].Prop = Prop;
			DeviceProp[CUDA_DEVICE_MIN_GFLOPS].DeviceId = i;
		}
	}

	int32 SelectedDeviceId = -1;
	switch (Mode)
	{
	case FIRST_PCI_BUS_ID:
		SelectedDeviceId = DeviceProp[CUDA_DEVICE_FIRST_PCI_BUS_ID].DeviceId;
		break;
	case LAST_PCI_BUS_ID:
		SelectedDeviceId = DeviceProp[CUDA_DEVICE_LAST_PCI_BUS_ID].DeviceId;
		break;
	case MAX_GFLOPS:
		SelectedDeviceId = DeviceProp[CUDA_DEVICE_MAX_GFLOPS].DeviceId;
		break;
	case MIN_GFLOPS:
		SelectedDeviceId = DeviceProp[CUDA_DEVICE_MIN_GFLOPS].DeviceId;
		break;
	case SPECIFIED_DEVICE_ID:
		SelectedDeviceId = DeviceProp[CUDA_DEVICE_SPECIFIED].DeviceId;
		break;
	default:
		check(false);
		break;
	}

	if (-1 == SelectedDeviceId)
	{
		goto lb_return;
	}

	if (cudaSetDevice(SelectedDeviceId) != cudaSuccess)
	{
		goto lb_return;
	}

	CudaSelectedDeviceId = SelectedDeviceId;

	cudaGetDeviceProperties(&Prop, SelectedDeviceId);

	strcpy_s(GpuInfo.szDeviceName, Prop.name);

	Prop.kernelExecTimeoutEnabled;
	uint32 SmPerMultiproc;
	if (Prop.major == 9999 && Prop.minor == 9999)
	{
		SmPerMultiproc = 1;
	}
	else
	{
		SmPerMultiproc = static_cast<uint32>(_ConvertSMVer2Cores(Prop.major, Prop.minor));
	}
	GpuInfo.sm_per_multiproc = static_cast<uint32>(SmPerMultiproc);
	GpuInfo.clock_rate = static_cast<uint32>(Prop.clockRate);
	GpuInfo.multiProcessorCount = static_cast<uint32>(Prop.multiProcessorCount);
	uint64 KFlops = static_cast<uint64>(static_cast<uint32>(Prop.multiProcessorCount)) * static_cast<uint64>(static_cast<uint32>(SmPerMultiproc)) * static_cast<uint64>(static_cast<uint32>(Prop.clockRate)) * 2;
	GpuInfo.TFlops = static_cast<float>(KFlops) / static_cast<float>(1024 * 1024 * 1024);
	bCanPrefetch = Prop.concurrentManagedAccess != 0;

	bResult = true;

lb_return:
	return bResult;
}

void AParticlesActor::InitializeGrid(int32 Size, float Spacing, float Jitter, int32 InNumParticles)
{
#if RENDER_INSTANCES
	UE_LOG(LogTemp, Warning, TEXT("InitializeGrid, ParticlesCount: %d / %d"), InNumParticles, ParticleInstancedMeshComponent->GetNumRenderInstances());
#endif
	FVector Scale(ParticleRadius / ParticleRenderRadius, ParticleRadius / ParticleRenderRadius, ParticleRadius / ParticleRenderRadius);

	for (int32 z = 0; z < Size; ++z)
	{
		for (int32 y = 0; y < Size; ++y)
		{
			for (int32 x = 0; x < Size; ++x)
			{
				int32 i = (z * Size * Size) + (y * Size) + x;

				if (i < InNumParticles)
				{
					switch (SphPlatform)
					{
					case ESphPlatform::E_CPU_SINGLE_THREAD:
					{
						FVector Vector((SupportRadius * x) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter,
							(SupportRadius * z) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter,
							(SupportRadius * y) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter);
#if RENDER_INSTANCES
						ParticleInstancedMeshComponent->UpdateInstanceTransform(i,
							FTransform(FRotator::ZeroRotator, Vector, FVector(ParticleRenderRadius / ParticleMeshRadius)),
							true,
							true,
							true);
#endif

						//Particles[i]->SetActorLocation(Vector);
						//Particles[i]->SetActorScale3D(Scale);
						//UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f)"),
						//	i,
						//	x, y, z,
						//	Vector.X, Vector.Y, Vector.Z);
					}
					break;
					case ESphPlatform::E_CPU_MULTIPLE_THREADS:
						break;
					case ESphPlatform::E_GPU_CUDA:
						HostPositions[i * 4] = (Spacing * x) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[i * 4 + 1] = (Spacing * y) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[i * 4 + 2] = (Spacing * z) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[i * 4 + 3] = 1.0f;

						HostVelocities[i * 4] = 0.0f;
						HostVelocities[i * 4 + 1] = 0.0f;
						HostVelocities[i * 4 + 2] = 0.0f;
						HostVelocities[i * 4 + 3] = 0.0f;

#if RENDER_INSTANCES
						{
							FVector Vector(HostPositions[i * 4] * ParticleRenderRadius / ParticleRadius,
								HostPositions[i * 4 + 2] * ParticleRenderRadius / ParticleRadius,
								(HostPositions[i * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
							bool Result = ParticleInstancedMeshComponent->UpdateInstanceTransform(i,
								FTransform(FRotator::ZeroRotator, Vector, FVector(ParticleRenderRadius / ParticleMeshRadius)),
								true);
							//UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f)"),
							//	i,
							//	x, y, z,
							//	HostPositions[i * 4], HostPositions[i * 4 + 1], HostPositions[i * 4 + 2]);
							UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f), %s"),
								i,
								x, y, z,
								Vector.X, Vector.Y, Vector.Z,
								Result ? *FString("True") : *FString("False"));
						}
#endif
						break;
					default:
						break;
					}
				}
			}
		}
	}
#if RENDER_INSTANCES
	UE_LOG(LogTemp, Warning, TEXT("InitGrid::Num Render Instances: %d, %d"), ParticleInstancedMeshComponent->GetNumRenderInstances(), ParticleInstancedMeshComponent->GetInstanceCount());
	ParticleInstancedMeshComponent->MarkRenderStateDirty();
#endif
}

void AParticlesActor::CleanUpCuda()
{
	VerifyCudaError(cudaDeviceReset());
}

void AParticlesActor::Destroy()
{
	if (HostPositions != nullptr)
	{
		FMemory::Free(HostPositions);
	}
	if (HostVelocities != nullptr)
	{
		FMemory::Free(HostVelocities);
	}
	if (HostForces != nullptr)
	{
		FMemory::Free(HostForces);
	}
	if (HostPressureForces != nullptr)
	{
		FMemory::Free(HostPressureForces);
	}
	if (HostViscosityForces != nullptr)
	{
		FMemory::Free(HostViscosityForces);
	}
	//if (HostSurfaceTensionForces != nullptr)
	//{
	//	FMemory::Free(HostSurfaceTensionForces);
	//}
	if (HostDensities != nullptr)
	{
		FMemory::Free(HostDensities);
	}
	if (HostPressures != nullptr)
	{
		FMemory::Free(HostPressures);
	}
	if (HostCellStarts != nullptr)
	{
		FMemory::Free(HostCellStarts);
	}
	if (HostCellEnds != nullptr)
	{
		FMemory::Free(HostCellEnds);
	}
	if (DeviceVelocities != nullptr)
	{
		cudaFree(DeviceVelocities);
	}
	if (DeviceDensities != nullptr)
	{
		cudaFree(DeviceDensities);
	}
	if (DeviceForces != nullptr)
	{
		cudaFree(DeviceForces);
	}
	if (DevicePressureForces != nullptr)
	{
		cudaFree(DevicePressureForces);
	}
	if (DeviceViscosityForces != nullptr)
	{
		cudaFree(DeviceViscosityForces);
	}
	//if (DeviceSurfaceTensionForces != nullptr)
	//{
	//	cudaFree(DeviceSurfaceTensionForces);
	//}
	if (DevicePressures != nullptr)
	{
		cudaFree(DevicePressures);
	}
	if (DeviceSortedPositions != nullptr)
	{
		cudaFree(DeviceSortedPositions);
	}
	if (DeviceSortedVelocities != nullptr)
	{
		cudaFree(DeviceSortedVelocities);
	}
	if (DeviceGridParticleHashes != nullptr)
	{
		cudaFree(DeviceGridParticleHashes);
	}
	if (DeviceGridParticleIndice != nullptr)
	{
		cudaFree(DeviceGridParticleIndice);
	}
	if (DeviceCellStarts != nullptr)
	{
		cudaFree(DeviceCellStarts);
	}
	if (DeviceCellEnds != nullptr)
	{
		cudaFree(DeviceCellEnds);
	}
	if (CudaPositionsVbo != nullptr)
	{
		cudaFree(CudaPositionsVbo);
	}
	if (DeviceMcPositions != nullptr)
	{
		cudaFree(DeviceMcPositions);
	}
	if (DeviceMcNormals != nullptr)
	{
		cudaFree(DeviceMcNormals);
	}
	if (HostMcVertices != nullptr)
	{
		cudaFreeHost(HostMcVertices);
	}
	if (HostMcNormals != nullptr)
	{
		cudaFreeHost(HostMcNormals);
	}
	CudaDestroyAllTextureObjects();
	if (DeviceEdgeTable != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceEdgeTable));
	}
	if (DeviceTriTable != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceTriTable));
	}
	if (DeviceNumVerticesTable != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceNumVerticesTable));
	}
	if (DeviceVoxelVertices != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVoxelVertices));
	}
	if (DeviceVoxelVerticesScan != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVoxelVerticesScan));
	}
	if (DeviceVoxelsOccupied != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVoxelsOccupied));
	}
	if (DeviceVoxelsOccupiedScan != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVoxelsOccupiedScan));
	}
	if (DeviceCompactedVoxelArray != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceCompactedVoxelArray));
	}
	if (DeviceVolumes != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVolumes));
	}
}

uint32 AParticlesActor::CalculateGridHash(int3 GridPosition)
{
	GridPosition.x = GridPosition.x & (SimulationParameters.GridSize.x - 1);
	GridPosition.y = GridPosition.y & (SimulationParameters.GridSize.y - 1);
	GridPosition.z = GridPosition.z & (SimulationParameters.GridSize.z - 1);

	return static_cast<uint32>(((GridPosition.z * SimulationParameters.GridSize.y) * SimulationParameters.GridSize.x) + (GridSize.y * SimulationParameters.GridSize.x) + GridPosition.x);
}

int3 AParticlesActor::CalculateGridPosition(float3 Position)
{
	int3 GridPosition;
	GridPosition.x = static_cast<int>(FMath::Floor((Position.x - SimulationParameters.WorldOrigin.x) / SimulationParameters.CellSize.x));
	GridPosition.y = static_cast<int>(FMath::Floor((Position.y - SimulationParameters.WorldOrigin.y) / SimulationParameters.CellSize.y));
	GridPosition.z = static_cast<int>(FMath::Floor((Position.z - SimulationParameters.WorldOrigin.z) / SimulationParameters.CellSize.z));
	return GridPosition;
}

void AParticlesActor::CalculateHash(uint32* GridParticleHashes, uint32* GridParticleIndices, float* Positions)
{
	switch (SphPlatform)
	{
	case ESphPlatform::E_CPU_SINGLE_THREAD:
		for (uint32 i = 0; i < NumParticles; ++i)
		{
			float4 Position = make_float4(Positions[i * 4], Positions[i * 4 + 1], Positions[i * 4 + 2], Positions[i * 4 + 3]);
			int3 GridPosition = CalculateGridPosition(make_float3(Position.x, Position.y, Position.z));
			uint32 Hash = CalculateGridHash(GridPosition);

			GridParticleHashes[i] = Hash;
			GridParticleIndices[i] = i;
		}
		break;
	case ESphPlatform::E_CPU_MULTIPLE_THREADS:
		break;
	case ESphPlatform::E_GPU_CUDA:
		break;
	default:
		break;
	}
}

float AParticlesActor::CalculatePoly6BySquaredDistance(float SquaredDistance)
{
	//return 315.0f * FMath::Pow(SupportRadiusSquared - SquaredDistance, 3.0f) / Poly6Denominator;
	return 4.0f * FMath::Pow(SupportRadiusSquared - SquaredDistance, 3.0f) / (PI * FMath::Pow(SupportRadiusSquared, 4.0f));
}

FVector AParticlesActor::CalculatePoly6Gradient(const FVector& Vector)
{
	//return -(945.0f * FMath::Pow(SupportRadiusSquared - Vector.SizeSquared(), 2.0f) * Vector) / (32.0f * PI * FMath::Pow(SupportRadius, 9.0f));
	return -(24.0f * FMath::Pow(SupportRadiusSquared - Vector.SizeSquared(), 2.0f) * Vector) / (PI * FMath::Pow(SupportRadiusSquared, 4.0f));
}

float3 AParticlesActor::CalculatePoly6Gradient(float3 Vector)
{
	float3 Result = Vector;
	Result *= -24.0f * FMath::Pow(SupportRadiusSquared - (Vector.x * Vector.x + Vector.y * Vector.y + Vector.z * Vector.z), 2.0f);
	Result /= (PI * FMath::Pow(SupportRadiusSquared, 4.0f));
	return Result;
}

FVector AParticlesActor::CalculateSpikyGradient(const FVector& Vector)
{
	float Distance = Vector.Size();
	float Offset = SupportRadius - Distance;
	//UE_LOG(LogTemp, Warning, TEXT("\t\tDistance=%f, Offset=%f, Vector=(%f, %f, %f), Norm=(%f, %f, %f), nom=(%f, %f, %f), denom=%f"), 
	//	Distance, Offset, 
	//	Vector.X, Vector.Y, Vector.Z, 
	//	Vector.GetSafeNormal().X, Vector.GetSafeNormal().Y, Vector.GetSafeNormal().Z,
	//	45.0f * Offset * Offset * Vector.GetSafeNormal().X, 45.0f * Offset * Offset * Vector.GetSafeNormal().Y, 45.0f * Offset * Offset * Vector.GetSafeNormal().Z,
	//	PI * FMath::Pow(SupportRadiusSquared, 3.0f));
	//return -(45.0f * Offset * Offset * Vector.GetSafeNormal() / (PI * FMath::Pow(SupportRadiusSquared, 3.0f)));
	return -(10.0f * Offset * Offset * Offset * Vector.GetSafeNormal() / (PI * FMath::Pow(SupportRadius, 5.0f)));
}

float3 AParticlesActor::CalculateSpikyGradient(float3 Vector)
{
	float Distance = FMath::Sqrt(Vector.x * Vector.x + Vector.y * Vector.y + Vector.z * Vector.z);
	float Offset = SupportRadius - Distance;

	float3 Result = normalize(Vector);
	Result *= -(10.0f * Offset * Offset * Offset / (PI * FMath::Pow(SupportRadius, 5.0f)));

	return Result;
}

float AParticlesActor::CalculateViscosityLaplacianByDistance(float Distance)
{
	//return (45.0f * (SupportRadius - Distance)) / (PI * FMath::Pow(SupportRadiusSquared, 3.0f));
	return (40.0f * (SupportRadius - Distance)) / (PI * FMath::Pow(SupportRadius, 5.0f));
}

void AParticlesActor::ComputeDensityAndPressure(float* OutDensities, float* OutPressures, float* SortedPositions, uint32* GridParticleIndices, uint32* CellStart, uint32* CellEnd, uint32 InNumParticles, uint32 InNumCells)
{
	for (uint32 i = 0; i < InNumParticles; ++i)
	{
		// read particle data from sorted arrays
		float3 IPosition = make_float3(SortedPositions[i * 4], SortedPositions[i * 4 + 1], SortedPositions[i * 4 + 2]);

		// get address in Grid

		// examine neighbouring cells
		float Density = 0.0f;

		for (uint32 j = 0; j < InNumParticles; ++j)
		{
			float3 JPosition = make_float3(SortedPositions[j * 4], SortedPositions[j * 4 + 1], SortedPositions[j * 4 + 2]);
			float3 Rij = make_float3(IPosition.x - JPosition.x, IPosition.y - JPosition.y, IPosition.z - JPosition.z);
			float SquaredDistance = Rij.x * Rij.x + Rij.y * Rij.y + Rij.z * Rij.z;

			if (SquaredDistance < SupportRadiusSquared)
			{
				Density += Mass * CalculatePoly6BySquaredDistance(SquaredDistance);
			}
		}
		OutDensities[i] = Density;
	}
}

void AParticlesActor::ComputeVelocities(float* OutVelocities, float* SortedPositions, float* SortedVelocities, float* Densities, float* Pressures, float DeltaTime, uint32* GridParticleIndices, uint32* CellStart, uint32* CellEnd, uint32 InNumParticles, uint32 NumCells)
{
	for (uint32 i = 0; i < InNumParticles; ++i)
	{
		// read particle data from sorted arrays
		float3 IPosition = make_float3(SortedPositions[i * 4], SortedPositions[i * 4 + 1], SortedPositions[i * 4 + 2]);
		float3 IVelocity = make_float3(SortedVelocities[i * 4], SortedVelocities[i * 4 + 1], SortedVelocities[i * 4 + 2]);
		float Density = Densities[i];
		float Pressure = Pressures[i];

		// get address in Grid

		// examine neighbouring cells
		float3 PressureForce = make_float3(0.0f, 0.0f, 0.0f);
		float3 ViscosityForce = make_float3(0.0f, 0.0f, 0.0f);

		for (uint32 j = 0; j < InNumParticles; ++j)
		{
			if (j != i)
			{
				float3 JPosition = make_float3(SortedPositions[j * 4], SortedPositions[j * 4 + 1], SortedPositions[j * 4 + 2]);
				float3 Rij = make_float3(IPosition.x - JPosition.x, IPosition.y - JPosition.y, IPosition.z - JPosition.z);
				float Distance = FMath::Sqrt(Rij.x * Rij.x + Rij.y * Rij.y + Rij.z * Rij.z);

				if (Distance < SupportRadius)
				{
					float3 JVelocity = make_float3(SortedVelocities[j * 4], SortedVelocities[j * 4 + 1], SortedVelocities[j * 4 + 2]);
					float3 Vij = make_float3(IVelocity.x - JVelocity.x, IVelocity.y - JVelocity.y, IVelocity.z - JVelocity.z);

					PressureForce -= Mass * ((Pressure / (Density * Density)) + (Pressures[j] / (Densities[j] * Densities[j]))) * CalculateSpikyGradient(Rij);
					ViscosityForce += (Mass / Densities[j]) * (dot(Vij, Rij) / (Distance * Distance + 0.01 * SupportRadiusSquared)) * CalculatePoly6Gradient(Rij);
				}
			}
		}
		PressureForce *= Density;
		ViscosityForce *= Viscosity * 10.0f;
		float3 ExternalForce = make_float3(Gravity.X, Gravity.Y, Gravity.Z) * Density;
		float3 TotalForce = PressureForce + ViscosityForce + ExternalForce;

		float4 OutVelocityOffset = make_float4((TotalForce / Density) * DeltaTime, 0.0f);
		OutVelocities[i * 4] = OutVelocityOffset.x;
		OutVelocities[i * 4 + 1] = OutVelocityOffset.y;
		OutVelocities[i * 4 + 2] = OutVelocityOffset.z;
		OutVelocities[i * 4 + 3] = OutVelocityOffset.w;
	}
}

void AParticlesActor::Integrate(float* Positions, float* Velocities, float DeltaTime, uint32 InNumParticles)
{
	FVector Scale(ParticleMeshRadius / ParticleRenderRadius, ParticleMeshRadius / ParticleRenderRadius, ParticleMeshRadius / ParticleRenderRadius);

	for (uint32 i = 0; i < InNumParticles; ++i)
	{
		float3 Position = make_float3(Positions[i * 4], Positions[i * 4 + 1], Positions[i * 4 + 2]);
		float3 Velocity = make_float3(Velocities[i * 4], Velocities[i * 4 + 1], Velocities[i * 4 + 2]);

		Position += Velocity * DeltaTime;

		if (Position.x > 1.0f - ParticleRadius)
		{
			Position.x = 1.0f - ParticleRadius;
			Velocity.x *= BoundaryDamping;
		}

		if (Position.x < -1.0f + ParticleRadius)
		{
			Position.x = -1.0f + ParticleRadius;
			Velocity.x *= BoundaryDamping;
		}

		if (Position.y > 1.0f - ParticleRadius)
		{
			Position.y = 1.0f - ParticleRadius;
			Velocity.y *= BoundaryDamping;
		}

		if (Position.z > 1.0f - ParticleRadius)
		{
			Position.z = 1.0f - ParticleRadius;
			Velocity.z *= BoundaryDamping;
		}

		if (Position.z < 0.0f + ParticleRadius)
		{
			Position.z = 0.0f + ParticleRadius;
			Velocity.z *= BoundaryDamping;
		}

		if (Position.y < -1.0f + ParticleRadius)
		{
			Position.y = -1.0f + ParticleRadius;
			Velocity.y *= BoundaryDamping;
		}

		Positions[i * 4] = Position.x;
		Positions[i * 4 + 1] = Position.y;
		Positions[i * 4 + 2] = Position.z;

		Velocities[i * 4] = Velocity.x;
		Velocities[i * 4 + 1] = Velocity.y;
		Velocities[i * 4 + 2] = Velocity.z;

		FVector VectorLocation(Position.x * ParticleRenderRadius / ParticleRadius, Position.y * ParticleRenderRadius / ParticleRadius, Position.z * ParticleRenderRadius / ParticleRadius);
#if RENDER_INSTANCES
		ParticleInstancedMeshComponent->UpdateInstanceTransform(i,
			FTransform(FRotator::ZeroRotator, VectorLocation, FVector(ParticleRenderRadius / ParticleMeshRadius)),
			true,
			true,
			true);
		ParticleInstancedMeshComponent->MarkRenderStateDirty();
#endif
		//Particles[i]->SetActorLocation(VectorLocation);
		//Particles[i]->SetActorScale3D(Scale);

		UE_LOG(LogTemp, Warning, TEXT("%d: Location=(%f, %f, %f)"), i, VectorLocation.X, VectorLocation.Y, VectorLocation.Z);
	}
	CudaIntegrateSystem(DevicePositions, DeviceVelocities, NumParticles);
}

void AParticlesActor::SortParticles(uint32* GridParticleHashes, uint32* GridParticleIndices)
{
}

void AParticlesActor::Reset()
{
	float Jitter = SimulationParameters.ParticleRadius * 0.01f;
	uint32 Size = static_cast<int32>(FMath::CeilToFloat(FMath::Pow(static_cast<float>(NumParticles), 1.0f / 3.0f)));
	UE_LOG(LogTemp, Warning, TEXT("Grid Size=%u"), Size);

	InitializeGrid(Size, SimulationParameters.SupportRadius, Jitter, NumParticles);
	//int32 p = 0;
	//int32 v = 0;

	//for (uint32 i = 0; i < NumParticles; i++)
	//{
	//	float point[3];
	//	point[0] = FMath::FRand();
	//	point[1] = FMath::FRand();
	//	point[2] = FMath::FRand();
	//	HostPositions[p++] = 2 * (point[0] - 0.5f);
	//	HostPositions[p++] = 2 * (point[1] - 0.5f);
	//	HostPositions[p++] = 2 * (point[2] - 0.5f);
	//	HostPositions[p++] = 1.0f; // radius
	//	HostVelocities[v++] = 0.0f;
	//	HostVelocities[v++] = 0.0f;
	//	HostVelocities[v++] = 0.0f;
	//	HostVelocities[v++] = 0.0f;
	//}

	cudaMemcpy(CudaPositionsVbo, HostPositions, sizeof(float) * NumParticles * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceVelocities, HostVelocities, sizeof(float) * NumParticles * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceForces, HostForces, sizeof(float) * NumParticles * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceDensities, HostDensities, sizeof(float) * NumParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(DevicePressures, HostPressures, sizeof(float) * NumParticles, cudaMemcpyHostToDevice);
}

void AParticlesActor::CreateIsosurface()
{
#if !RENDER_INSTANCES
	ParticleProceduralMeshComponent->ClearAllMeshSections();

	cudaMemcpy(HostMcVertices, DeviceMcPositions, sizeof(float) * 4 * NumMaxVertice, cudaMemcpyDeviceToHost);
	cudaMemcpy(HostMcNormals, DeviceMcNormals, sizeof(float) * 4 * NumMaxVertice, cudaMemcpyDeviceToHost);

	Vertices.Empty();
	Triangles.Empty();
	Uv0.Empty();

	Vertices.Reserve(NumMaxVertice);
	Triangles.Reserve(NumMaxVertice);
	Uv0.Reserve(NumMaxVertice);

	//UE_LOG(LogTemp, Warning, TEXT("=============================START============================="));
	//for (uint32 i = 0; i < NumTotalVertice; ++i)
	//{
	//	UE_LOG(LogTemp, Warning, TEXT("[%u]: (%f, %f, %f, %f)"), i, HostMcVertices[i].x, HostMcVertices[i].y, HostMcVertices[i].z, HostMcVertices[i].w);
	//}
	//UE_LOG(LogTemp, Warning, TEXT("==============================END=============================="));

	//UE_LOG(LogTemp, Warning, TEXT("Total Verts: %u, %u"), NumTotalVertice, NumTotalVertice % 3u);
	for (uint32 i = 0; i < NumTotalVertice / 3u; ++i)
	{
		//UE_LOG(LogTemp, Warning, TEXT("[%u] vertice: (%f, %f, %f)"), i, HostMcVertices[3 * i].x, HostMcVertices[3 * i].y, HostMcVertices[3 * i].z);

		Vertices.Add(FVector(-HostMcVertices[3 * i].x,
			-HostMcVertices[3 * i].y,
			(-HostMcVertices[3 * i].z + 1.0f)) * ParticleRenderRadius / ParticleRadius);

		Vertices.Add(FVector(-HostMcVertices[3 * i + 1].x,
			-HostMcVertices[3 * i + 1].y,
			(-HostMcVertices[3 * i + 1].z + 1.0f)) * ParticleRenderRadius / ParticleRadius);

		Vertices.Add(FVector(-HostMcVertices[3 * i + 2].x,
			-HostMcVertices[3 * i + 2].y,
			(-HostMcVertices[3 * i + 2].z + 1.0f)) * ParticleRenderRadius / ParticleRadius);

		Triangles.Add(static_cast<int32>(3 * i + 2));
		Triangles.Add(static_cast<int32>(3 * i + 1));
		Triangles.Add(static_cast<int32>(3 * i + 0));

		Uv0.Add(FVector2D(-HostMcVertices[3 * i].x / 2.0f, -HostMcVertices[3 * i].y / 2.0f));
		Uv0.Add(FVector2D(-HostMcVertices[3 * i + 1].x / 2.0f, -HostMcVertices[3 * i + 1].y / 2.0f));
		Uv0.Add(FVector2D(-HostMcVertices[3 * i + 2].x / 2.0f, -HostMcVertices[3 * i + 2].y / 2.0f));
		Normals.Add(FVector(-HostMcNormals[3 * i].x, -HostMcNormals[3 * i].y, -HostMcNormals[3 * i].z));
		Normals.Add(FVector(-HostMcNormals[3 * i + 1].x, -HostMcNormals[3 * i + 1].y, -HostMcNormals[3 * i + 1].z));
		Normals.Add(FVector(-HostMcNormals[3 * i + 2].x, -HostMcNormals[3 * i + 2].y, -HostMcNormals[3 * i + 2].z));
		//UE_LOG(LogTemp, Warning, TEXT("(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)"), 
		//	Vertices[3 * i].X, Vertices[3 * i].Y, Vertices[3 * i].Z,
		//	Vertices[3 * i + 1].X, Vertices[3 * i + 1].Y, Vertices[3 * i + 1].Z, 
		//	Vertices[3 * i + 2].X, Vertices[3 * i + 2].Y, Vertices[3 * i + 2].Z);
		//UE_LOG(LogTemp, Warning, TEXT("(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)"),
		//	Vertices[3 * i].X, Vertices[3 * i].Y, Vertices[3 * i].Z,
		//	Vertices[3 * i + 1].X, Vertices[3 * i + 1].Y, Vertices[3 * i + 1].Z,
		//	Vertices[3 * i + 2].X, Vertices[3 * i + 2].Y, Vertices[3 * i + 2].Z);
	}
	//UE_LOG(LogTemp, Warning, TEXT("Total Vertices: %u, %d"), NumTotalVertice, Vertices.Num());
	ParticleProceduralMeshComponent->CreateMeshSection(0, Vertices, Triangles, Normals, Uv0, VertexColors, Tangents, false);
	//ParticleProceduralMeshComponent->ContainsPhysicsTriMeshData(true);
	if (Material != nullptr)
	{
		ParticleProceduralMeshComponent->SetMaterial(0, Material);
	}
#endif
}

void AParticlesActor::Tick(float DeltaTime)
{
	DevicePositions = CudaPositionsVbo;
	SimulationParameters.DeltaTime = FMath::Min(DeltaTime * 0.2f, CustomDeltaTime);
	UE_LOG(LogTemp, Warning, TEXT("Tick: %f ~ %f"), DeltaTime, SimulationParameters.DeltaTime);

	switch (SphPlatform)
	{
	case ESphPlatform::E_CPU_SINGLE_THREAD:
		ComputeDensityAndPressure(HostDensities, HostPressures, HostPositions, DeviceGridParticleIndice, HostCellStarts, HostCellEnds, NumParticles, NumGridCells);
		ComputeVelocities(HostVelocities, HostPositions, HostVelocities, HostDensities, HostPressures, DeltaTime, DeviceGridParticleIndice, HostCellStarts, HostCellEnds, NumParticles, NumGridCells);
		Integrate(HostPositions, HostVelocities, DeltaTime, NumParticles);
		break;
	case ESphPlatform::E_CPU_MULTIPLE_THREADS:
		break;
	case ESphPlatform::E_GPU_CUDA:
		// update constants
		CudaSetParameters(&SimulationParameters);

		// calculate Grid hash
		CudaCalculateHashes(DeviceGridParticleHashes, DeviceGridParticleIndice, DevicePositions, NumParticles);

		// sort particles based on hash
		CudaSortParticles(DeviceGridParticleHashes, DeviceGridParticleIndice, NumParticles);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		CudaReorderDataAndFindCellStart(DeviceCellStarts,
										DeviceCellEnds,
										DeviceSortedPositions,
										DeviceSortedVelocities,
										DeviceGridParticleHashes,
										DeviceGridParticleIndice,
										DevicePositions,
										DeviceVelocities,
										NumParticles,
										NumGridCells);

		//cudaMemcpy(HostPositions, DevicePositions, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		//cudaMemcpy(HostVelocities, DeviceVelocities, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);

		//for (uint32 i = 0; i < NumParticles; ++i)
		//{
		//	UE_LOG(LogTemp, Warning, TEXT("%u: BEFORE TICK Position=(%f, %f, %f), Velocity=(%f, %f, %f)"),
		//		i,
		//		HostPositions[i * 4], HostPositions[i * 4 + 1], HostPositions[i * 4 + 2],
		//		HostVelocities[i * 4], HostVelocities[i * 4 + 1], HostVelocities[i * 4 + 2]);
		//}

		CudaComputeDensitiesAndPressures(DeviceDensities,
										 DevicePressures,
										 DeviceSortedPositions,
										 DeviceGridParticleIndice,
										 DeviceCellStarts,
										 DeviceCellEnds,
										 NumParticles);

		//cudaMemcpy(HostDensities, DeviceDensities, sizeof(float) * NumParticles, cudaMemcpyDeviceToHost);
		////cudaMemcpy(HostPressures, DevicePressures, sizeof(float) * NumParticles, cudaMemcpyDeviceToHost);
		//for (uint32 i = 0; i < NumParticles; ++i)
		//{
		//	//UE_LOG(LogTemp, Warning, TEXT("%u: Density=%f, Pressure=%f"), i, HostDensities[i], HostPressures[i]);
		//	if (FMath::IsNaN(HostDensities[i]))
		//	{
		//		UE_LOG(LogTemp, Warning, TEXT("%u: Density=%f !!"), i, HostDensities[i]);
		//	}
		//	UE_LOG(LogTemp, Warning, TEXT("%u: Density=%f"), i, HostDensities[i]);
		//}

		CudaComputeAllForcesAndVelocities(DeviceVelocities,
										  DeviceForces,
										  DevicePressureForces,
										  DeviceViscosityForces,
										  DeviceSortedPositions,
										  DeviceSortedVelocities,
										  DeviceDensities,
										  DevicePressures,
   										  DeviceGridParticleIndice,
										  DeviceCellStarts,
										  DeviceCellEnds,
										  NumParticles);
		//CudaComputeForcesAndVelocities(DeviceVelocities,
		//							   DeviceForces,
		//							   DeviceSortedPositions,
		//							   DeviceSortedVelocities,
		//							   DeviceDensities,
		//							   DevicePressures,
  // 									   DeviceGridParticleIndice,
		//							   DeviceCellStarts,
		//							   DeviceCellEnds,
		//							   NumParticles);

		//cudaMemcpy(HostVelocities, DeviceVelocities, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(HostForces, DeviceForces, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(HostPressureForces, DevicePressureForces, sizeof(float) * 4u * NumParticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(HostViscosityForces, DeviceViscosityForces, sizeof(float) * 4u * NumParticles, cudaMemcpyDeviceToHost);
		//////cudaMemcpy(HostSurfaceTensionForces, DeviceSurfaceTensionForces, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		for (uint32 i = FMath::Max(NumParticles - 10u, 0u); i < NumParticles; ++i)
		{
		//	//UE_LOG(LogTemp, Warning, TEXT("%u: Velocity=(%f, %f, %f), Forces=(%f, %f, %f)"),
		//	//	i,
		//	//	HostVelocities[i * 4], HostVelocities[i * 4 + 1], HostVelocities[i * 4 + 2],
		//	//	HostForces[i * 4], HostForces[i * 4 + 1], HostForces[i * 4 + 2]);
		//	//if (FMath::IsNaN(HostPressureForces[i * 4]) ||
		//	//	FMath::IsNaN(HostPressureForces[i * 4 + 1]) ||
		//	//	FMath::IsNaN(HostPressureForces[i * 4 + 2]) ||
		//	//	FMath::IsNaN(HostViscosityForces[i * 4]) ||
		//	//	FMath::IsNaN(HostViscosityForces[i * 4 + 1]) ||
		//	//	FMath::IsNaN(HostViscosityForces[i * 4 + 2]) ||
		//	//	FMath::IsNaN(HostSurfaceTensionForces[i * 4]) ||
		//	//	FMath::IsNaN(HostSurfaceTensionForces[i * 4 + 1]) ||
		//	//	FMath::IsNaN(HostSurfaceTensionForces[i * 4 + 2]))
		//	//{
		//	//	//UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f), V=(%f, %f, %f), ST=(%f, %f, %f) !!"), i,
		//	//	//	HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2],
		//	//	//	HostViscosityForces[i * 4], HostViscosityForces[i * 4 + 1], HostViscosityForces[i * 4 + 2],
		//	//	//	HostSurfaceTensionForces[i * 4], HostSurfaceTensionForces[i * 4 + 1], HostSurfaceTensionForces[i * 4 + 2]);
		//	//	UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f) !!"), i,
		//	//		HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2]);
		//	//}
		//	//UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f), V=(%f, %f, %f), ST=(%f, %f, %f)"), i,
		//	//	HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2],
		//	//	HostViscosityForces[i * 4], HostViscosityForces[i * 4 + 1], HostViscosityForces[i * 4 + 2],
		//	//	HostSurfaceTensionForces[i * 4], HostSurfaceTensionForces[i * 4 + 1], HostSurfaceTensionForces[i * 4 + 2]);
		//	UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f) !!"), i,
		//		HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2]);
			//UE_LOG(LogTemp, Warning, TEXT("%u: Velocity=(%f, %f, %f), Forces=(%f, %f, %f)"),
			//	i,
			//	HostVelocities[i * 4], HostVelocities[i * 4 + 1], HostVelocities[i * 4 + 2],
			//	HostForces[i * 4], HostForces[i * 4 + 1], HostForces[i * 4 + 2]);
			UE_LOG(LogTemp, Warning, TEXT("[%u]: F=(%f, %f, %f) P=(%f, %f, %f), V=(%f, %f, %f)"), i,
				HostForces[i * 4], HostForces[i * 4 + 1], HostForces[i * 4 + 2],
				HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2],
				HostViscosityForces[i * 4], HostViscosityForces[i * 4 + 1], HostViscosityForces[i * 4 + 2]);
		}

		CudaIntegrateSystem(DevicePositions, DeviceVelocities, NumParticles);

#if RENDER_INSTANCES
		cudaMemcpy(HostPositions, DevicePositions, sizeof(float) * 4u * NumParticles, cudaMemcpyDeviceToHost);

		//UE_LOG(LogTemp, Warning, TEXT("Tick::NumParticles: %u"), NumParticles);
		for (uint32 i = 0; i < NumParticles; ++i)
		{
			FVector VectorLocation(HostPositions[i * 4] * ParticleRenderRadius / ParticleRadius,
				HostPositions[i * 4 + 2] * ParticleRenderRadius / ParticleRadius,
				(HostPositions[i * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
			//	//Particles[i]->SetActorLocation(VectorLocation);
			ParticleInstancedMeshComponent->UpdateInstanceTransform(i,
				FTransform(FRotator::ZeroRotator, VectorLocation, FVector(ParticleRenderRadius / ParticleMeshRadius)),
				true);
			//UE_LOG(LogTemp, Warning, TEXT("%u: Location=(%f, %f, %f)"), i, VectorLocation.X, VectorLocation.Y, VectorLocation.Z);
		}
#endif
		break;
	default:
		break;
	}

#if !RENDER_INSTANCES
	// marching cubes algorithm
	int Threads = 128;
	dim3 Grid(NumVoxels / Threads, 1, 1);

	// get around maximum Grid size of 65535 in each dimension
	if (Grid.x > 65535)
	{
		Grid.y = Grid.x / 32768;
		Grid.x = 32768;
	}

	// calculate number of vertices need per voxel
	CudaLaunchClassifyVoxels(Grid, 
							 Threads,
							 DeviceVoxelVertices, 
							 DeviceVoxelsOccupied, 
							 DeviceVolumes,
							 GridSize, 
							 GridSizeShift, 
							 GridSizeMask,
							 NumVoxels, 
							 VoxelSize, 
							 IsoValue, 
							 DeviceSortedPositions, 
							 DeviceCellStarts, 
							 DeviceCellEnds);

#if DEBUG_BUFFERS
	printf("voxelVerts:\n");
	dumpBuffer(d_voxelVerts, numVoxels, sizeof(uint));
#endif

#if SKIP_EMPTY_VOXELS
	// scan voxel occupied array
	CudaThrustScanWrapper(DeviceVoxelsOccupiedScan, 
						  DeviceVoxelsOccupied, 
						  NumVoxels);

#if DEBUG_BUFFERS
	printf("voxelOccupiedScan:\n");
	dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint LastElement;
		uint LastScanElement;
		checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(&LastElement),
								   reinterpret_cast<void*>(DeviceVoxelsOccupied + NumVoxels - 1),
								   sizeof(uint), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(&LastScanElement),
								   reinterpret_cast<void*>(DeviceVoxelsOccupiedScan + NumVoxels - 1),
								   sizeof(uint), 
								   cudaMemcpyDeviceToHost));
		NumActiveVoxels = LastElement + LastScanElement;
		//UE_LOG(LogTemp, Warning, TEXT("NonEmpty Voxels after Classification: %u"), NumActiveVoxels);
	}

	if (NumActiveVoxels == 0)
	{
		// return if there are no full voxels
		NumTotalVertice = 0;
		return;
	}

	// compact voxel index array
	CudaLaunchCompactVoxels(Grid, 
							Threads, 
							DeviceCompactedVoxelArray, 
							DeviceVoxelsOccupied, 
							DeviceVoxelsOccupiedScan, 
							NumVoxels);
	getLastCudaError("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS

	// scan voxel vertex count array
	CudaThrustScanWrapper(DeviceVoxelVerticesScan, DeviceVoxelVertices, NumVoxels);

#if DEBUG_BUFFERS
	printf("voxelVertsScan:\n");
	dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

	// readback total number of vertices
	{
		uint LastElement;
		uint LastScanElement;
		checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(&LastElement),
								   reinterpret_cast<void*>(DeviceVoxelVertices + NumVoxels - 1),
								   sizeof(uint), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(&LastScanElement),
								   reinterpret_cast<void*>(DeviceVoxelVerticesScan + NumVoxels - 1),
								   sizeof(uint), 
								   cudaMemcpyDeviceToHost));
		NumTotalVertice = LastElement + LastScanElement;
		//UE_LOG(LogTemp, Warning, TEXT("Number of vertices after compacting: %u"), NumTotalVertice);
	}

#if SKIP_EMPTY_VOXELS
	dim3 Grid2(static_cast<int>(ceil(NumActiveVoxels / static_cast<float>(NTHREADS))), 1, 1);
#else
	dim3 Grid2(static_cast<int>(ceil(NumVoxels / static_cast<float>(NTHREADS))), 1, 1);
#endif

	while (Grid2.x > 65535)
	{
		Grid2.x /= 2;
		Grid2.y *= 2;
	}

#if SAMPLE_VOLUME
	CudaLaunchGenerateTriangles2(Grid2,
								 NTHREADS,
								 DeviceMcPositions, 
								 DeviceMcNormals,
								 DeviceCompactedVoxelArray,
								 DeviceVoxelVerticesScan, 
								 DeviceVolumes,
								 GridSize,
								 GridSizeShift, 
								 GridSizeMask,
								 VoxelSize, 
								 IsoValue, 
								 NumActiveVoxels,
								 NumMaxVertice);
#else
	CudaLaunchGenerateTriangles(Grid2, 
								NTHREADS, 
								DeviceMcPositions, 
								DeviceMcNormals,
								DeviceCompactedVoxelArray,
								DeviceVoxelVerticesScan,
								GridSize, 
								GridSizeShift, 
								GridSizeMask,
								VoxelSize, 
								IsoValue, 
								NumActiveVoxels,
								NumMaxVertice, 
								DeviceSortedPositions, 
								DeviceCellStarts, 
								DeviceCellEnds);
#endif
#endif

#if RENDER_INSTANCES
	ParticleInstancedMeshComponent->MarkRenderStateDirty();
#else
	CreateIsosurface();
#endif
}

