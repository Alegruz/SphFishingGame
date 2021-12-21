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
	, GridSortBits(18u)	// increase this for larger grids
	, GridSize(make_uint3(GRID_SIZE, GRID_SIZE, GRID_SIZE))
	, NumGridCells(GridSize.x * GridSize.y * GridSize.z)
	, GridSizeLog2(make_uint3(GRID_SIZE_LOG2, GRID_SIZE_LOG2, GRID_SIZE_LOG2))
	, McGridSize(make_uint3(1u << GridSizeLog2.x, 1u << GridSizeLog2.y, 1u << GridSizeLog2.z))
	, GridSizeMask(make_uint3(McGridSize.x - 1, McGridSize.y - 1, McGridSize.z - 1))
	, GridSizeShift(make_uint3(0, GridSizeLog2.x, GridSizeLog2.x + GridSizeLog2.y))
	, VoxelSize(make_float3(2.0f / McGridSize.x, 2.0f / McGridSize.y, 2.0f / McGridSize.z))
	, NumVoxels(McGridSize.x* McGridSize.y* McGridSize.z)
	, NumMaxVertices(McGridSize.x* McGridSize.y * McGridSize.z * 64u)
	, IsoValue(SupportRadiusSquared * 1.1f)
	, DeviceIsoValue(IsoValue)
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	ParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>("ParticleInstancedMeshComponent");
	BoundaryParticleInstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>("BoundaryParticleInstancedMeshComponent");
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

	NumFishes = static_cast<int32>(NumBoundaryParticles / NUM_BOUNDARY_PARTICLES_PER_FISH);
	// set simulation parameters
	SimulationParameters.GridSize = GridSize;
	SimulationParameters.McGridSize = GridSize;
	SimulationParameters.NumCells = NumGridCells;
	SimulationParameters.NumVoxels = NumVoxels;
	SimulationParameters.NumBodies = NumParticles;

	SimulationParameters.ParticleRadius = ParticleRadius;

	SimulationParameters.WorldOrigin = make_float3(-1.0f, -1.0f, -1.0f);

	SimulationParameters.BoundaryDamping = BoundaryDamping;

	SimulationParameters.ParticleMass = FluidParticleMass;
	SimulationParameters.BoundaryParticleMass = BoundaryParticleMass;
	SimulationParameters.SupportRadius = SupportRadius;
	SimulationParameters.SupportRadiusSquared = SupportRadiusSquared;

	const float CellSize = SimulationParameters.SupportRadius;  // cell size equal to particle diameter
	SimulationParameters.CellSize = make_float3(CellSize, CellSize, CellSize);
	SimulationParameters.McCellSize = VoxelSize;

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

	SimulationParameters.XScaleFactor = GetActorScale().X;
	SimulationParameters.YScaleFactor = GetActorScale().Z;
	SimulationParameters.ZScaleFactor = GetActorScale().Y;

	SimulationParameters.MarchingCubesNeighborSearchDepth = MarchingCubesNeighborSearchDepth;
	//SimulationParameters.XScaleFactor = 0.5f;
	//SimulationParameters.YScaleFactor = 1.0f;
	//SimulationParameters.ZScaleFactor = 1.0f;

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

	if (bIsInitialized)
	{
		Destroy();
	}
	Initialize();
	Reset();
}

void AParticlesActor::Initialize()
{
	if (bIsInitialized)
	{
		return;
	}

	FishesCoordinates.Reserve(NumFishes);
	FishesVelocities.Reserve(NumFishes);

	uint32 MemorySize = sizeof(float) * 4u * NumParticles;

	// allocate host storage
	HostPositions = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostVelocities = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	//HostPressureForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	//HostViscosityForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	//HostSurfaceTensionForces = reinterpret_cast<float*>(FMemory::Malloc(MemorySize));
	HostDensities = reinterpret_cast<float*>(FMemory::Malloc(sizeof(float) * NumParticles));
	HostPressures = reinterpret_cast<float*>(FMemory::Malloc(sizeof(float) * NumParticles));

	FMemory::Memzero(reinterpret_cast<void*>(HostPositions), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostVelocities), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostForces), MemorySize);
	//FMemory::Memzero(reinterpret_cast<void*>(HostPressureForces), MemorySize);
	//FMemory::Memzero(reinterpret_cast<void*>(HostViscosityForces), MemorySize);
	//FMemory::Memzero(reinterpret_cast<void*>(HostSurfaceTensionForces), MemorySize);
	FMemory::Memzero(reinterpret_cast<void*>(HostDensities), sizeof(float) * NumParticles);
	FMemory::Memzero(reinterpret_cast<void*>(HostPressures), sizeof(float) * NumParticles);

	HostGridParticleHashes = reinterpret_cast<uint32*>(FMemory::Malloc(sizeof(uint32) * NumParticles));
	FMemory::Memzero(reinterpret_cast<void*>(HostGridParticleHashes), sizeof(uint32) * NumParticles);

	HostGridParticleIndices = reinterpret_cast<uint32*>(FMemory::Malloc(sizeof(uint32) * NumParticles));
	FMemory::Memzero(reinterpret_cast<void*>(HostGridParticleIndices), sizeof(uint32) * NumParticles);

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
		//cudaMalloc(reinterpret_cast<void**>(&DevicePressureForces), MemorySize);
		//cudaMalloc(reinterpret_cast<void**>(&DeviceViscosityForces), MemorySize);
		//cudaMalloc(reinterpret_cast<void**>(&DeviceSurfaceTensionForces), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceDensities), sizeof(float) * NumParticles);
		cudaMalloc(reinterpret_cast<void**>(&DevicePressures), sizeof(float) * NumParticles);

		cudaMalloc(reinterpret_cast<void**>(&DeviceSortedPositions), MemorySize);
		cudaMalloc(reinterpret_cast<void**>(&DeviceSortedVelocities), MemorySize);

		cudaMalloc(reinterpret_cast<void**>(&DeviceGridParticleHashes), sizeof(uint32) * NumParticles);
		cudaMalloc(reinterpret_cast<void**>(&DeviceGridParticleIndices), sizeof(uint32) * NumParticles);

		cudaMalloc(reinterpret_cast<void**>(&DeviceCellStarts), sizeof(uint32) * NumGridCells);
		cudaMalloc(reinterpret_cast<void**>(&DeviceCellEnds), sizeof(uint32) * NumGridCells);

		CudaSetParameters(&SimulationParameters);
	}

#if !RENDER_INSTANCES
	UE_LOG(LogTemp, Warning, TEXT("MC GRID SIZE: %u x %u x %u = %u voxels"), McGridSize.x, McGridSize.y, McGridSize.z, NumVoxels);
	UE_LOG(LogTemp, Warning, TEXT("MAX VERTICE: %u"), NumMaxVertices);
	
	HostMcCellStarts = reinterpret_cast<uint32*>(FMemory::Malloc(sizeof(uint32) * NumVoxels));
	FMemory::Memzero(reinterpret_cast<void*>(HostMcCellStarts), sizeof(uint32) * NumVoxels);

	HostMcCellEnds = reinterpret_cast<uint32*>(FMemory::Malloc(sizeof(uint32) * NumVoxels));
	FMemory::Memzero(reinterpret_cast<void*>(HostMcCellEnds), sizeof(uint32) * NumVoxels);

	cudaMalloc(reinterpret_cast<void**>(&DeviceMcSortedPositions), MemorySize);

	//cudaMalloc(reinterpret_cast<void**>(&DeviceMcGridParticleHashes), sizeof(uint32) * NumParticles);
	//cudaMalloc(reinterpret_cast<void**>(&DeviceMcGridParticleIndices), sizeof(uint32) * NumParticles);

	//cudaMalloc(reinterpret_cast<void**>(&DeviceMcCellStarts), sizeof(uint32) * NumVoxels);
	//cudaMalloc(reinterpret_cast<void**>(&DeviceMcCellEnds), sizeof(uint32) * NumVoxels);

	cudaMalloc(reinterpret_cast<void**>(&DeviceMcPositions), NumMaxVertices * sizeof(float) * 4);
	cudaMalloc(reinterpret_cast<void**>(&DeviceMcNormals), NumMaxVertices * sizeof(float) * 4);
	cudaMallocHost(reinterpret_cast<void**>(&HostMcVerticess), NumMaxVertices * sizeof(float) * 4);
	cudaMallocHost(reinterpret_cast<void**>(&HostMcNormals), NumMaxVertices * sizeof(float) * 4);

	// allocate textures
	CudaAllocateTextures(&DeviceEdgeTable, &DeviceTriTable, &DeviceNumVerticessTable);

	// allocate device memory
	MemorySize = sizeof(uint32) * NumVoxels;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelVerticess), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelVerticessScan), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelsOccupied), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceVoxelsOccupiedScan), MemorySize));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&DeviceCompactedVoxelArray), MemorySize));

	Verticess.Reserve(NumMaxVertices);
	Triangles.Reserve(NumMaxVertices);
	Uv0.Reserve(NumMaxVertices);
#else
	ParticleInstancedMeshComponent->PreAllocateInstancesMemory(NumFluidParticles);
	UE_LOG(LogTemp, Warning, TEXT("Initialize with %u Particles"), NumParticles);
	for (uint32 Index = 0; Index < NumFluidParticles; ++Index)
	{
		FVector Location(0.0f, 0.0f, 0.0f);
		int32 Result = ParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(ParticleRenderRadius / ParticleMeshRadius)));
		UE_LOG(LogTemp, Warning, TEXT("Adding Instance... %d / %d"), ParticleInstancedMeshComponent->GetInstanceCount(), Result);
	}
	UE_LOG(LogTemp, Warning, TEXT("Initialize::Num Render Instances: %d, %d"), ParticleInstancedMeshComponent->GetNumRenderInstances(), BoundaryParticleInstancedMeshComponent->GetNumRenderInstances());
#endif
	BoundaryParticleInstancedMeshComponent->PreAllocateInstancesMemory(NumBoundaryParticles);
	for (uint32 Index = 0; Index < NumBoundaryParticles; ++Index)
	{
		FVector Location(0.0f, 0.0f, 0.0f);
		int32 Result = BoundaryParticleInstancedMeshComponent->AddInstance(FTransform(FRotator::ZeroRotator, Location, FVector(ParticleRenderRadius / ParticleMeshRadius)));
		UE_LOG(LogTemp, Warning, TEXT("Adding Boundary Instance... %d / %d"), BoundaryParticleInstancedMeshComponent->GetInstanceCount(), Result);
	}

	bIsInitialized = true;
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

void AParticlesActor::InitializeGrid(int32 Size, float Spacing, float Jitter, int32 InNumFluidParticles, int32 InBoundaryParticles)
{
#if RENDER_INSTANCES
	UE_LOG(LogTemp, Warning, TEXT("InitializeGrid, ParticlesCount: %d / %d"), InNumFluidParticles, ParticleInstancedMeshComponent->GetNumRenderInstances());
#endif
	FVector Scale(ParticleRadius / ParticleRenderRadius, ParticleRadius / ParticleRenderRadius, ParticleRadius / ParticleRenderRadius);

	for (int32 z = 0; z < Size; ++z)
	{
		for (int32 y = 0; y < Size; ++y)
		{
			for (int32 x = 0; x < Size; ++x)
			{
				int32 Index = (z * Size * Size) + (y * Size) + x;

				if (Index < static_cast<int32>(NumRenderingFluidParticles) / 2)
				{
					switch (SphPlatform)
					{
					case ESphPlatform::E_CPU_SINGLE_THREAD:
					{
						FVector Vector((SupportRadius * x) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter,
							(SupportRadius * z) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter,
							(SupportRadius * y) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter);
#if RENDER_INSTANCES
						ParticleInstancedMeshComponent->UpdateInstanceTransform(Index,
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
						HostPositions[Index * 4] = (Spacing * x) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[Index * 4 + 1] = (Spacing * y) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[Index * 4 + 2] = (Spacing * z) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[Index * 4 + 3] = 1.0f;

						HostVelocities[Index * 4] = 0.0f;
						HostVelocities[Index * 4 + 1] = 0.0f;
						HostVelocities[Index * 4 + 2] = 0.0f;
						HostVelocities[Index * 4 + 3] = 0.0f;

#if RENDER_INSTANCES
						{
							FVector Vector(HostPositions[Index * 4] * ParticleRenderRadius / ParticleRadius,
								HostPositions[Index * 4 + 2] * ParticleRenderRadius / ParticleRadius,
								(HostPositions[Index * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
							bool Result = ParticleInstancedMeshComponent->UpdateInstanceTransform(Index,
								FTransform(FRotator::ZeroRotator, Vector, FVector(ParticleRenderRadius / ParticleMeshRadius)),
								true);
							//UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f)"),
							//	i,
							//	x, y, z,
							//	HostPositions[i * 4], HostPositions[i * 4 + 1], HostPositions[i * 4 + 2]);
							UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f), %s"),
								Index,
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

	for (int32 z = 0; z < Size; ++z)
	{
		for (int32 y = 0; y < Size; ++y)
		{
			for (int32 x = 0; x < Size; ++x)
			{
				int32 Index = (z * Size * Size) + (y * Size) + x;

				if (Index < static_cast<int32>(NumRenderingFluidParticles) / 2)
				{
					switch (SphPlatform)
					{
					case ESphPlatform::E_CPU_SINGLE_THREAD:
					{
						FVector Vector((SupportRadius * x) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter,
							(SupportRadius * z) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter,
							(SupportRadius * y) + ParticleRadius - 1.0f * (ParticleRenderRadius / ParticleRadius) + (FMath::FRand() * 2.0f - 1.0f) * Jitter);
#if RENDER_INSTANCES
						ParticleInstancedMeshComponent->UpdateInstanceTransform(Index,
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
						HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4] = -((Spacing * x) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter);
						HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4 + 1] = (Spacing * y) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter;
						HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4 + 2] = -((Spacing * z) + SimulationParameters.ParticleRadius - 1.0f + (FMath::FRand() * 2.0f - 1.0f) * Jitter);
						HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4 + 3] = 1.0f;

						HostVelocities[(Index + NumRenderingFluidParticles / 2u) * 4] = 0.0f;
						HostVelocities[(Index + NumRenderingFluidParticles / 2u) * 4 + 1] = 0.0f;
						HostVelocities[(Index + NumRenderingFluidParticles / 2u) * 4 + 2] = 0.0f;
						HostVelocities[(Index + NumRenderingFluidParticles / 2u) * 4 + 3] = 0.0f;

#if RENDER_INSTANCES
						{
							FVector Vector(HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4] * ParticleRenderRadius / ParticleRadius,
								HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4 + 2] * ParticleRenderRadius / ParticleRadius,
								(HostPositions[(Index + NumRenderingFluidParticles / 2u) * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
							bool Result = ParticleInstancedMeshComponent->UpdateInstanceTransform((Index + NumRenderingFluidParticles / 2u),
								FTransform(FRotator::ZeroRotator, Vector, FVector(ParticleRenderRadius / ParticleMeshRadius)),
								true);
							//UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f)"),
							//	i,
							//	x, y, z,
							//	HostPositions[i * 4], HostPositions[i * 4 + 1], HostPositions[i * 4 + 2]);
							UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f), %s"),
								Index,
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

	for (uint32 Index = NumRenderingFluidParticles; Index < NumFluidParticles; ++Index)
	{
		HostPositions[Index * 4] = 0.0f;
		HostPositions[Index * 4 + 1] = 0.0f;
		HostPositions[Index * 4 + 2] = 0.0f;
		HostPositions[Index * 4 + 3] = 1.0f;

		HostVelocities[Index * 4] = 0.0f;
		HostVelocities[Index * 4 + 1] = 0.0f;
		HostVelocities[Index * 4 + 2] = 0.0f;
		HostVelocities[Index * 4 + 3] = 0.0f;

#if RENDER_INSTANCES
		{
			FVector Vector(HostPositions[Index * 4] * ParticleRenderRadius / ParticleRadius,
				HostPositions[Index * 4 + 2] * ParticleRenderRadius / ParticleRadius,
				(HostPositions[Index * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
			bool Result = ParticleInstancedMeshComponent->UpdateInstanceTransform(Index,
				FTransform(FRotator::ZeroRotator, Vector, FVector(0.0f)),
				true);
			//UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f)"),
			//	i,
			//	x, y, z,
			//	HostPositions[i * 4], HostPositions[i * 4 + 1], HostPositions[i * 4 + 2]);
		}
#endif
	}

	//uint32 Thickness = ((NumBoundaryParticles + (static_cast<uint32>((1.0f / ParticleRadius) * (1.0f / (2.0f * ParticleRadius))) - 1u)) / static_cast<uint32>((1.0f / ParticleRadius) * (1.0f / (2.0f * ParticleRadius))));

	//for (uint32 y = 0u; y < Thickness; ++y)
	//{
	//	for (uint32 z = 0u; z < static_cast<uint32>((1.0f / (2.0f * ParticleRadius))); ++z)
	//	{
	//		for (uint32 x = 0u; x < static_cast<uint32>((1.0f / ParticleRadius)); ++x)
	//		{
	//			uint32 Index = (z * Size * Size) + (y * Size) + x;

	//			if (Index < NumBoundaryParticles)
	//			{
	//				HostPositions[(NumMaxFluidParticles + Index) * 4] = (SimulationParameters.ParticleRadius * x) + SimulationParameters.ParticleRadius - 1.0f;
	//				HostPositions[(NumMaxFluidParticles + Index) * 4 + 1] = (SimulationParameters.ParticleRadius * z) + SimulationParameters.ParticleRadius - 1.0f;
	//				HostPositions[(NumMaxFluidParticles + Index) * 4 + 2] = 0.25f + y * SimulationParameters.ParticleRadius;
	//				HostPositions[(NumMaxFluidParticles + Index) * 4 + 3] = 1.0f;

	//				HostVelocities[(NumMaxFluidParticles + Index) * 4] = 0.0f;
	//				HostVelocities[(NumMaxFluidParticles + Index) * 4 + 1] = 0.0f;
	//				HostVelocities[(NumMaxFluidParticles + Index) * 4 + 2] = 0.0f;
	//				HostVelocities[(NumMaxFluidParticles + Index) * 4 + 3] = 0.0f;

	//				{
	//					FVector Vector(HostPositions[(NumMaxFluidParticles + Index) * 4] * ParticleRenderRadius / ParticleRadius,
	//						HostPositions[(NumMaxFluidParticles + Index) * 4 + 2] * ParticleRenderRadius / ParticleRadius,
	//						(HostPositions[(NumMaxFluidParticles + Index) * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
	//					bool Result = BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(Index,
	//						FTransform(FRotator::ZeroRotator, Vector, FVector(ParticleRenderRadius / ParticleMeshRadius)),
	//						true);
	//					//UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f)"),
	//					//	i,
	//					//	x, y, z,
	//					//	HostPositions[i * 4], HostPositions[i * 4 + 1], HostPositions[i * 4 + 2]);
	//					UE_LOG(LogTemp, Warning, TEXT("%d: InitGrid(%d, %d, %d)::Position=(%f, %f, %f), %s"),
	//						NumMaxFluidParticles + Index,
	//						x, y, z,
	//						Vector.X, Vector.Y, Vector.Z,
	//						Result ? *FString("True") : *FString("False"));
	//				}
	//			}
	//		}
	//	}
	//}

#if RENDER_INSTANCES
	UE_LOG(LogTemp, Warning, TEXT("InitGrid::Num Render Instances: %d, %d"), ParticleInstancedMeshComponent->GetNumRenderInstances(), ParticleInstancedMeshComponent->GetInstanceCount());
	ParticleInstancedMeshComponent->MarkRenderStateDirty();
	BoundaryParticleInstancedMeshComponent->MarkRenderStateDirty();
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
	//if (HostPressureForces != nullptr)
	//{
	//	FMemory::Free(HostPressureForces);
	//}
	//if (HostViscosityForces != nullptr)
	//{
	//	FMemory::Free(HostViscosityForces);
	//}
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
	if (HostGridParticleHashes != nullptr)
	{
		FMemory::Free(HostGridParticleHashes);
	}
	if (HostGridParticleIndices != nullptr)
	{
		FMemory::Free(HostGridParticleIndices);
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
	//if (DevicePressureForces != nullptr)
	//{
	//	cudaFree(DevicePressureForces);
	//}
	//if (DeviceViscosityForces != nullptr)
	//{
	//	cudaFree(DeviceViscosityForces);
	//}
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
	if (DeviceGridParticleIndices != nullptr)
	{
		cudaFree(DeviceGridParticleIndices);
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
	if (HostMcCellStarts != nullptr)
	{
		FMemory::Free(HostMcCellStarts);
	}
	if (HostMcCellEnds != nullptr)
	{
		FMemory::Free(HostMcCellEnds);
	}
	if (DeviceMcSortedPositions != nullptr)
	{
		cudaFree(DeviceMcSortedPositions);
	}
	//if (DeviceMcGridParticleHashes != nullptr)
	//{
	//	cudaFree(DeviceMcGridParticleHashes);
	//}
	//if (DeviceMcGridParticleIndices != nullptr)
	//{
	//	cudaFree(DeviceMcGridParticleIndices);
	//}
	//if (DeviceMcCellStarts != nullptr)
	//{
	//	cudaFree(DeviceMcCellStarts);
	//}
	//if (DeviceMcCellEnds != nullptr)
	//{
	//	cudaFree(DeviceMcCellEnds);
	//}
	if (DeviceMcPositions != nullptr)
	{
		cudaFree(DeviceMcPositions);
	}
	if (DeviceMcNormals != nullptr)
	{
		cudaFree(DeviceMcNormals);
	}
	if (HostMcVerticess != nullptr)
	{
		cudaFreeHost(HostMcVerticess);
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
	if (DeviceNumVerticessTable != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceNumVerticessTable));
	}
	if (DeviceVoxelVerticess != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVoxelVerticess));
	}
	if (DeviceVoxelVerticessScan != nullptr)
	{
		checkCudaErrors(cudaFree(DeviceVoxelVerticessScan));
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
	bIsInitialized = false;
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

void AParticlesActor::CalculateHash(uint32* GridParticleHashes, uint32* GridParticleIndicess, float* Positions)
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
			GridParticleIndicess[i] = i;
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

void AParticlesActor::AddSphere()
{
	int32 BallRadius = 2;
	float Tr = SimulationParameters.ParticleRadius + (SimulationParameters.ParticleRadius * 2.0f) * static_cast<float>(BallRadius);
	float Position[4] = { -1.0f + Tr + (FMath::FRand() / RAND_MAX) * (2.0f - Tr * 2.0f), 
						  1.0f - Tr,
						  -1.0f + Tr + (FMath::FRand() / RAND_MAX) * (2.0f - Tr * 2.0f),
						  0.0f };
	float Velocity[4] = { 0.0f, };
	//psystem->AddSphere(0, position, velocity, gBallRadius, particleRadius * 2.0f);
	uint32 Index = 0u;
	uint32 Start = 0u;

	for (int Z = -BallRadius; Z <= BallRadius; ++Z)
	{
		for (int Y = -BallRadius; Y <= BallRadius; ++Y)
		{
			for (int X = -BallRadius; X <= BallRadius; ++X)
			{
				float DeltaX = static_cast<float>(X) * SimulationParameters.ParticleRadius * 2.0f;
				float DeltaY = static_cast<float>(Y) * SimulationParameters.ParticleRadius * 2.0f;
				float DeltaZ = static_cast<float>(Z) * SimulationParameters.ParticleRadius * 2.0f;
				float Length = FMath::Sqrt(DeltaX * DeltaX + DeltaY * DeltaY + DeltaZ * DeltaZ);
				float Jitter = SimulationParameters.ParticleRadius * 0.01f;

				if ((Length <= SimulationParameters.ParticleRadius * 2.0f * BallRadius) && (Index < NumFluidParticles))
				{
					HostPositions[Index * 4] = Position[0] + DeltaX + ((FMath::FRand() / RAND_MAX) * 2.0f - 1.0f) * Jitter;
					HostPositions[Index * 4 + 1] = Position[1] + DeltaY + ((FMath::FRand() / RAND_MAX) * 2.0f - 1.0f) * Jitter;
					HostPositions[Index * 4 + 2] = Position[2] + DeltaZ + ((FMath::FRand() / RAND_MAX) * 2.0f - 1.0f) * Jitter;
					HostPositions[Index * 4 + 3] = Position[3];

					HostVelocities[Index * 4] = Velocity[0];
					HostVelocities[Index * 4 + 1] = Velocity[1];
					HostVelocities[Index * 4 + 2] = Velocity[2];
					HostVelocities[Index * 4 + 3] = Velocity[3];
					++Index;
				}
			}
		}
	}

	//SetArray(POSITION, mHostPositions, start, index);
	//SetArray(VELOCITY, mHostVelocities, start, index);
	CudaCopyArrayToDevice(CudaPositionsVbo, HostPositions, Start * 4u * sizeof(float), Index * 4u * sizeof(float));
	CudaCopyArrayToDevice(DeviceVelocities, HostVelocities, Start * 4u * sizeof(float), Index * 4u * sizeof(float));
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

void AParticlesActor::ComputeDensityAndPressure(float* OutDensities, float* OutPressures, float* SortedPositions, uint32* GridParticleIndicess, uint32* CellStart, uint32* CellEnd, uint32 InNumParticles, uint32 InNumCells)
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
				Density += FluidParticleMass * CalculatePoly6BySquaredDistance(SquaredDistance);
			}
		}
		OutDensities[i] = Density;
	}
}

void AParticlesActor::ComputeVelocities(float* OutVelocities, float* SortedPositions, float* SortedVelocities, float* Densities, float* Pressures, float DeltaTime, uint32* GridParticleIndicess, uint32* CellStart, uint32* CellEnd, uint32 InNumParticles, uint32 NumCells)
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

					PressureForce -= FluidParticleMass * ((Pressure / (Density * Density)) + (Pressures[j] / (Densities[j] * Densities[j]))) * CalculateSpikyGradient(Rij);
					ViscosityForce += (FluidParticleMass / Densities[j]) * (dot(Vij, Rij) / (Distance * Distance + 0.01 * SupportRadiusSquared)) * CalculatePoly6Gradient(Rij);
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

	for (uint32 Index = 0; Index < InNumParticles; ++Index)
	{
		float3 Position = make_float3(Positions[Index * 4], Positions[Index * 4 + 1], Positions[Index * 4 + 2]);
		float3 Velocity = make_float3(Velocities[Index * 4], Velocities[Index * 4 + 1], Velocities[Index * 4 + 2]);

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

		Positions[Index * 4] = Position.x;
		Positions[Index * 4 + 1] = Position.y;
		Positions[Index * 4 + 2] = Position.z;

		Velocities[Index * 4] = Velocity.x;
		Velocities[Index * 4 + 1] = Velocity.y;
		Velocities[Index * 4 + 2] = Velocity.z;

		FVector VectorLocation(Position.x * ParticleRenderRadius / ParticleRadius, Position.y * ParticleRenderRadius / ParticleRadius, Position.z * ParticleRenderRadius / ParticleRadius);
#if RENDER_INSTANCES
		ParticleInstancedMeshComponent->UpdateInstanceTransform(Index,
			FTransform(FRotator::ZeroRotator, VectorLocation, FVector(ParticleRenderRadius / ParticleMeshRadius)),
			true,
			true,
			true);
		ParticleInstancedMeshComponent->MarkRenderStateDirty();
#endif
		//Particles[i]->SetActorLocation(VectorLocation);
		//Particles[i]->SetActorScale3D(Scale);

		UE_LOG(LogTemp, Warning, TEXT("%d: Location=(%f, %f, %f)"), Index, VectorLocation.X, VectorLocation.Y, VectorLocation.Z);
	}
	CudaIntegrateSystem(DevicePositions, DeviceVelocities, NumParticles);
}

void AParticlesActor::SortParticles(uint32* GridParticleHashes, uint32* GridParticleIndicess)
{
}

void AParticlesActor::Reset()
{
	float Jitter = SimulationParameters.ParticleRadius * 0.01f;
	uint32 Size = static_cast<int32>(FMath::CeilToFloat(FMath::Pow(static_cast<float>(NumParticles), 1.0f / 3.0f)));
	UE_LOG(LogTemp, Warning, TEXT("Grid Size=%u"), Size);

	InitializeGrid(Size, SimulationParameters.SupportRadius, Jitter, NumFluidParticles, NumBoundaryParticles);
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

	cudaMemcpy(HostMcVerticess, DeviceMcPositions, sizeof(float) * 4 * NumMaxVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(HostMcNormals, DeviceMcNormals, sizeof(float) * 4 * NumMaxVertices, cudaMemcpyDeviceToHost);

	Verticess.Empty();
	Triangles.Empty();
	Uv0.Empty();

	Verticess.Reserve(NumMaxVertices);
	Triangles.Reserve(NumMaxVertices);
	Uv0.Reserve(NumMaxVertices);

	//UE_LOG(LogTemp, Warning, TEXT("=============================START============================="));
	//for (uint32 i = 0; i < NumTotalVertices; ++i)
	//{
	//	UE_LOG(LogTemp, Warning, TEXT("[%u]: (%f, %f, %f, %f)"), i, HostMcVerticess[i].x, HostMcVerticess[i].y, HostMcVerticess[i].z, HostMcVerticess[i].w);
	//}
	//UE_LOG(LogTemp, Warning, TEXT("==============================END=============================="));

	//UE_LOG(LogTemp, Warning, TEXT("Total Verts: %u, %u"), NumTotalVertices, NumTotalVertices % 3u);
	for (uint32 i = 0; i < NumTotalVertices / 3u; ++i)
	{
		//UE_LOG(LogTemp, Warning, TEXT("[%u] vertice: (%f, %f, %f)"), i, HostMcVerticess[3 * i].x, HostMcVerticess[3 * i].y, HostMcVerticess[3 * i].z);

		Verticess.Add(FVector(-HostMcVerticess[3 * i].x,
			-HostMcVerticess[3 * i].y,
			(-HostMcVerticess[3 * i].z + 1.0f)) * ParticleRenderRadius / ParticleRadius);

		Verticess.Add(FVector(-HostMcVerticess[3 * i + 1].x,
			-HostMcVerticess[3 * i + 1].y,
			(-HostMcVerticess[3 * i + 1].z + 1.0f)) * ParticleRenderRadius / ParticleRadius);

		Verticess.Add(FVector(-HostMcVerticess[3 * i + 2].x,
			-HostMcVerticess[3 * i + 2].y,
			(-HostMcVerticess[3 * i + 2].z + 1.0f)) * ParticleRenderRadius / ParticleRadius);

		Triangles.Add(static_cast<int32>(3 * i + 2));
		Triangles.Add(static_cast<int32>(3 * i + 1));
		Triangles.Add(static_cast<int32>(3 * i + 0));

		Normals.Add(FVector(-HostMcNormals[3 * i].x, -HostMcNormals[3 * i].y, -HostMcNormals[3 * i].z));
		FVector Normal = Normals.Last();
		Normals.Add(FVector(-HostMcNormals[3 * i + 1].x, -HostMcNormals[3 * i + 1].y, -HostMcNormals[3 * i + 1].z));
		Normal += Normals.Last();
		Normals.Add(FVector(-HostMcNormals[3 * i + 2].x, -HostMcNormals[3 * i + 2].y, -HostMcNormals[3 * i + 2].z));
		Normal += Normals.Last();
		Normal.Normalize();

		if (FVector::DotProduct(Normal, FVector(0.0f, 0.0f, 1.0f)) < 0.1f)
		{
			if (FVector::DotProduct(Normal, FVector(0.0f, 1.0f, 0.0f)) < 0.1f)
			{

				Uv0.Add(FVector2D(-HostMcVerticess[3 * i].x / 2.0f, -HostMcVerticess[3 * i].z / 2.0f));
				Uv0.Add(FVector2D(-HostMcVerticess[3 * i + 1].x / 2.0f, -HostMcVerticess[3 * i + 1].z / 2.0f));
				Uv0.Add(FVector2D(-HostMcVerticess[3 * i + 2].x / 2.0f, -HostMcVerticess[3 * i + 2].z / 2.0f));
			}
			else
			{

				Uv0.Add(FVector2D(-HostMcVerticess[3 * i].y / 2.0f, -HostMcVerticess[3 * i].z / 2.0f));
				Uv0.Add(FVector2D(-HostMcVerticess[3 * i + 1].y / 2.0f, -HostMcVerticess[3 * i + 1].z / 2.0f));
				Uv0.Add(FVector2D(-HostMcVerticess[3 * i + 2].y / 2.0f, -HostMcVerticess[3 * i + 2].z / 2.0f));
			}
		}
		else
		{
			Uv0.Add(FVector2D(-HostMcVerticess[3 * i].x / 2.0f, -HostMcVerticess[3 * i].y / 2.0f));
			Uv0.Add(FVector2D(-HostMcVerticess[3 * i + 1].x / 2.0f, -HostMcVerticess[3 * i + 1].y / 2.0f));
			Uv0.Add(FVector2D(-HostMcVerticess[3 * i + 2].x / 2.0f, -HostMcVerticess[3 * i + 2].y / 2.0f));
		}
		//UE_LOG(LogTemp, Warning, TEXT("(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)"), 
		//	Verticess[3 * i].X, Verticess[3 * i].Y, Verticess[3 * i].Z,
		//	Verticess[3 * i + 1].X, Verticess[3 * i + 1].Y, Verticess[3 * i + 1].Z, 
		//	Verticess[3 * i + 2].X, Verticess[3 * i + 2].Y, Verticess[3 * i + 2].Z);
		//UE_LOG(LogTemp, Warning, TEXT("(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)"),
		//	Verticess[3 * i].X, Verticess[3 * i].Y, Verticess[3 * i].Z,
		//	Verticess[3 * i + 1].X, Verticess[3 * i + 1].Y, Verticess[3 * i + 1].Z,
		//	Verticess[3 * i + 2].X, Verticess[3 * i + 2].Y, Verticess[3 * i + 2].Z);
	}
	//UE_LOG(LogTemp, Warning, TEXT("Total Verticess: %u, %d"), NumTotalVertices, Verticess.Num());
	ParticleProceduralMeshComponent->CreateMeshSection(0, Verticess, Triangles, Normals, Uv0, VertexColors, Tangents, false);
	//ParticleProceduralMeshComponent->ContainsPhysicsTriMeshData(true);
	if (Material != nullptr)
	{
		ParticleProceduralMeshComponent->SetMaterial(0, Material);
	}
#endif
}

void AParticlesActor::Tick(float DeltaTime)
{
	SimulationParameters.DeltaTime = FMath::Min(DeltaTime * 0.1f, CustomDeltaTime);
	//UE_LOG(LogTemp, Warning, TEXT("Tick: %f ~ %f"), DeltaTime, SimulationParameters.DeltaTime);

	UpdateFishesLocation();

	int32 Index = NumMaxFluidParticles;
	//UE_LOG(LogTemp, Warning, TEXT("NumFishes: %d"), NumFishes);
	for (int32 FishIndex = 0; FishIndex < NumFishes; ++FishIndex)
	{	
		//HostPositions[Index * 4] = (FishesCoordinates[FishIndex].X) * ParticleRadius / ParticleRenderRadius;
		//HostPositions[Index * 4 + 1] = (FishesCoordinates[FishIndex].Z) * ParticleRadius / ParticleRenderRadius - 1.0f;
		//HostPositions[Index * 4 + 2] = (FishesCoordinates[FishIndex].Y) * ParticleRadius / ParticleRenderRadius;
		//HostPositions[Index * 4 + 3] = 1.0f;
		//HostVelocities[Index * 4] = (FishesVelocities[FishIndex].X) * ParticleRadius / ParticleRenderRadius;
		//HostVelocities[Index * 4 + 1] = (FishesVelocities[FishIndex].Z) * ParticleRadius / ParticleRenderRadius;
		//HostVelocities[Index * 4 + 2] = (FishesVelocities[FishIndex].Y) * ParticleRadius / ParticleRenderRadius;
		//HostVelocities[Index * 4 + 3] = 0.0f;
		//++Index;
		for (int32 Z = -1; Z <= 1; ++Z)
		{
			for (int32 Y = -1; Y <= 1; ++Y)
			{
				for (int32 X = -1; X <= 1; ++X)
				{
					if (X == 0 || Y == 0 || Z == 0)
					{
						continue;
					}
					if (Index > static_cast<int32>(NumParticles))
					{
						goto break_loop;
					}
					HostPositions[Index * 4] = (FishesCoordinates[FishIndex].X) * ParticleRadius / ParticleRenderRadius + X * (SimulationParameters.ParticleRadius / 2.0f);
					HostPositions[Index * 4 + 1] = (FishesCoordinates[FishIndex].Z) * ParticleRadius / ParticleRenderRadius + Z * (SimulationParameters.ParticleRadius / 2.0f) - 1.0f;
					HostPositions[Index * 4 + 2] = (FishesCoordinates[FishIndex].Y) * ParticleRadius / ParticleRenderRadius + Y * (SimulationParameters.ParticleRadius / 2.0f);
					HostPositions[Index * 4 + 3] = 1.0f;
					HostVelocities[Index * 4] = (FishesVelocities[FishIndex].X) * ParticleRadius / ParticleRenderRadius;
					HostVelocities[Index * 4 + 1] = (FishesVelocities[FishIndex].Z) * ParticleRadius / ParticleRenderRadius;
					HostVelocities[Index * 4 + 2] = (FishesVelocities[FishIndex].Y) * ParticleRadius / ParticleRenderRadius;
					HostVelocities[Index * 4 + 3] = 0.0f;
					++Index;
					//UE_LOG(LogTemp, Warning, TEXT(""))
				}
			}
		}
	break_loop:;
	}
	
	static int32 Counter = 0;
	if (Counter % 3 == 0 && NumRenderingFluidParticles <= NumMaxFluidParticles)
	{
		//UE_LOG(LogTemp, Warning, TEXT("Counter: %d, NumRenderingFluidParticles: %u"), Counter, NumRenderingFluidParticles);
		for (int Y = -1; Y <= 1; ++Y)
		{
			for (int X = -1; X <= 1; ++X)
			{
				if (NumRenderingFluidParticles >= NumMaxFluidParticles)
				{
					goto break_additional_fluid_loop;
				}

				if (X * X + Y * Y > 1)
				{
					continue;
				}

				HostPositions[NumRenderingFluidParticles * 4] = X * 2.0f * ParticleRadius;
				HostPositions[NumRenderingFluidParticles * 4 + 1] = Y * 2.0f * ParticleRadius;
				HostPositions[NumRenderingFluidParticles * 4 + 2] = 1.0f;
				HostPositions[NumRenderingFluidParticles * 4 + 3] = 1.0f;
				++NumRenderingFluidParticles;

				HostPositions[NumRenderingFluidParticles * 4] = X * 2.0f * ParticleRadius;
				HostPositions[NumRenderingFluidParticles * 4 + 1] = Y * 2.0f * ParticleRadius;
				HostPositions[NumRenderingFluidParticles * 4 + 2] = -1.0f;
				HostPositions[NumRenderingFluidParticles * 4 + 3] = 1.0f;
				++NumRenderingFluidParticles;
			}
		}
	break_additional_fluid_loop:;
	}
	++Counter;
	cudaMemcpy(CudaPositionsVbo, HostPositions, sizeof(float) * NumParticles * 4, cudaMemcpyHostToDevice);
	//cudaMemcpy(CudaPositionsVbo + sizeof(float) * 4 * NumMaxFluidParticles,
	//	HostPositions + sizeof(float) * 4 * NumMaxFluidParticles,
	//	sizeof(float) * 4 * NumBoundaryParticles,
	//	cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceVelocities + sizeof(float) * 4 * NumMaxFluidParticles,
		HostVelocities + sizeof(float) * 4 * NumMaxFluidParticles,
		sizeof(float) * 4 * NumBoundaryParticles,
		cudaMemcpyHostToDevice);
	DevicePositions = CudaPositionsVbo;

	switch (SphPlatform)
	{
	case ESphPlatform::E_CPU_SINGLE_THREAD:
		ComputeDensityAndPressure(HostDensities, HostPressures, HostPositions, DeviceGridParticleIndices, HostCellStarts, HostCellEnds, NumParticles, NumGridCells);
		ComputeVelocities(HostVelocities, HostPositions, HostVelocities, HostDensities, HostPressures, DeltaTime, DeviceGridParticleIndices, HostCellStarts, HostCellEnds, NumParticles, NumGridCells);
		Integrate(HostPositions, HostVelocities, DeltaTime, NumParticles);
		break;
	case ESphPlatform::E_CPU_MULTIPLE_THREADS:
		break;
	case ESphPlatform::E_GPU_CUDA:
		// update constants
		CudaSetParameters(&SimulationParameters);

		// calculate Grid hash
		CudaCalculateHashes(DeviceGridParticleHashes, DeviceGridParticleIndices, DevicePositions, NumParticles);

		//cudaMemcpy(HostGridParticleHashes, DeviceGridParticleHashes, sizeof(uint32) * NumParticles, cudaMemcpyDeviceToHost);
		//cudaMemcpy(HostGridParticleIndices, DeviceGridParticleIndices, sizeof(uint32) * NumParticles, cudaMemcpyDeviceToHost);
		//for (uint32 i = 0u; i < NumParticles; ++i)
		//{
		//	UE_LOG(LogTemp, Warning, TEXT("[%u] hash: %u, index: %u"), i, HostGridParticleHashes[i], HostGridParticleIndices[i]);
		//}

		// sort particles based on hash
		CudaSortParticles(DeviceGridParticleHashes, DeviceGridParticleIndices, NumParticles);

		// reorder particle arrays into sorted order and
		// find start and end of each cell
		CudaReorderDataAndFindCellStart(DeviceCellStarts,
										DeviceCellEnds,
										DeviceSortedPositions,
										DeviceSortedVelocities,
										DeviceGridParticleHashes,
										DeviceGridParticleIndices,
										DevicePositions,
										DeviceVelocities,
										NumParticles,
										NumGridCells);

		//cudaMemcpy(HostCellStarts, DeviceCellStarts, sizeof(uint32) * NumGridCells, cudaMemcpyDeviceToHost);
		//cudaMemcpy(HostCellEnds, DeviceCellEnds, sizeof(uint32) * NumGridCells, cudaMemcpyDeviceToHost);
		//for (uint32 i = 0u; i < NumGridCells; ++i)
		//{
		//	UE_LOG(LogTemp, Warning, TEXT("[%u] start: %u, end: %u"), i, HostCellStarts[i], HostCellEnds[i]);
		//}

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
										 DeviceGridParticleIndices,
										 DeviceCellStarts,
										 DeviceCellEnds,
										 NumBoundaryParticles,
										 NumFluidParticles,
										 NumRenderingFluidParticles,
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

		//CudaComputeAllForcesAndVelocities(DeviceVelocities,
		//								  DeviceForces,
		//								  DevicePressureForces,
		//								  DeviceViscosityForces,
		//								  DeviceSortedPositions,
		//								  DeviceSortedVelocities,
		//								  DeviceDensities,
		//								  DevicePressures,
  // 										  DeviceGridParticleIndices,
		//								  DeviceCellStarts,
		//								  DeviceCellEnds,
		//							      NumFluidParticles,
		//								  NumParticles);
		CudaComputeForcesAndVelocities(DeviceVelocities,
									   DeviceForces,
									   DeviceSortedPositions,
									   DeviceSortedVelocities,
									   DeviceDensities,
									   DevicePressures,
   									   DeviceGridParticleIndices,
									   DeviceCellStarts,
									   DeviceCellEnds,
									   NumFluidParticles,
									   NumRenderingFluidParticles,
									   NumParticles);

		//cudaMemcpy(HostVelocities, DeviceVelocities, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		//cudaMemcpy(HostForces, DeviceForces, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		//cudaMemcpy(HostPressureForces, DevicePressureForces, sizeof(float) * 4u * NumParticles, cudaMemcpyDeviceToHost);
		//cudaMemcpy(HostViscosityForces, DeviceViscosityForces, sizeof(float) * 4u * NumParticles, cudaMemcpyDeviceToHost);
		////////cudaMemcpy(HostSurfaceTensionForces, DeviceSurfaceTensionForces, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
		//for (uint32 i = FMath::Max(NumParticles - 10u, 0u); i < NumParticles; ++i)
		//{
		////	//UE_LOG(LogTemp, Warning, TEXT("%u: Velocity=(%f, %f, %f), Forces=(%f, %f, %f)"),
		////	//	i,
		////	//	HostVelocities[i * 4], HostVelocities[i * 4 + 1], HostVelocities[i * 4 + 2],
		////	//	HostForces[i * 4], HostForces[i * 4 + 1], HostForces[i * 4 + 2]);
		////	//if (FMath::IsNaN(HostPressureForces[i * 4]) ||
		////	//	FMath::IsNaN(HostPressureForces[i * 4 + 1]) ||
		////	//	FMath::IsNaN(HostPressureForces[i * 4 + 2]) ||
		////	//	FMath::IsNaN(HostViscosityForces[i * 4]) ||
		////	//	FMath::IsNaN(HostViscosityForces[i * 4 + 1]) ||
		////	//	FMath::IsNaN(HostViscosityForces[i * 4 + 2]) ||
		////	//	FMath::IsNaN(HostSurfaceTensionForces[i * 4]) ||
		////	//	FMath::IsNaN(HostSurfaceTensionForces[i * 4 + 1]) ||
		////	//	FMath::IsNaN(HostSurfaceTensionForces[i * 4 + 2]))
		////	//{
		////	//	//UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f), V=(%f, %f, %f), ST=(%f, %f, %f) !!"), i,
		////	//	//	HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2],
		////	//	//	HostViscosityForces[i * 4], HostViscosityForces[i * 4 + 1], HostViscosityForces[i * 4 + 2],
		////	//	//	HostSurfaceTensionForces[i * 4], HostSurfaceTensionForces[i * 4 + 1], HostSurfaceTensionForces[i * 4 + 2]);
		////	//	UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f) !!"), i,
		////	//		HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2]);
		////	//}
		////	//UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f), V=(%f, %f, %f), ST=(%f, %f, %f)"), i,
		////	//	HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2],
		////	//	HostViscosityForces[i * 4], HostViscosityForces[i * 4 + 1], HostViscosityForces[i * 4 + 2],
		////	//	HostSurfaceTensionForces[i * 4], HostSurfaceTensionForces[i * 4 + 1], HostSurfaceTensionForces[i * 4 + 2]);
		////	UE_LOG(LogTemp, Warning, TEXT("[%u]: P=(%f, %f, %f) !!"), i,
		////		HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2]);
		//	//UE_LOG(LogTemp, Warning, TEXT("%u: Velocity=(%f, %f, %f), Forces=(%f, %f, %f)"),
		//	//	i,
		//	//	HostVelocities[i * 4], HostVelocities[i * 4 + 1], HostVelocities[i * 4 + 2],
		//	//	HostForces[i * 4], HostForces[i * 4 + 1], HostForces[i * 4 + 2]);
		//	UE_LOG(LogTemp, Warning, TEXT("[%u]: F=(%f, %f, %f) P=(%f, %f, %f), V=(%f, %f, %f)"), i,
		//		HostForces[i * 4], HostForces[i * 4 + 1], HostForces[i * 4 + 2],
		//		HostPressureForces[i * 4], HostPressureForces[i * 4 + 1], HostPressureForces[i * 4 + 2],
		//		HostViscosityForces[i * 4], HostViscosityForces[i * 4 + 1], HostViscosityForces[i * 4 + 2]);
		//}
	
		CudaIntegrateSystem(DevicePositions, DeviceVelocities, NumRenderingFluidParticles);

		//UE_LOG(LogTemp, Warning, TEXT("num particles: %d"), NumParticles);
		//cudaMemcpy(HostPositions, DevicePositions, sizeof(float) * 4u * NumParticles, cudaMemcpyDeviceToHost);
		cudaMemcpy(HostPositions + sizeof(float) * 4 * NumMaxFluidParticles, 
			DevicePositions + sizeof(float) * 4 * NumMaxFluidParticles, 
			sizeof(float) * 4u * NumBoundaryParticles, 
			cudaMemcpyDeviceToHost);
		for (uint32 BoundaryParticleIndex = NumFluidParticles; BoundaryParticleIndex < NumParticles; ++BoundaryParticleIndex)
		{
			//UE_LOG(LogTemp, Warning, TEXT("[%u]: Boundary Position=(%f, %f, %f)"), Index, HostPositions[Index * 4], HostPositions[Index * 4 + 1], HostPositions[Index * 4 + 2], HostPositions[Index * 4 + 3]);
			FVector VectorLocation(HostPositions[BoundaryParticleIndex * 4] * ParticleRenderRadius / ParticleRadius,
				HostPositions[BoundaryParticleIndex * 4 + 2] * ParticleRenderRadius / ParticleRadius,
				(HostPositions[BoundaryParticleIndex * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
			////	//Particles[i]->SetActorLocation(VectorLocation);
			BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(BoundaryParticleIndex - NumFluidParticles,
				FTransform(FRotator::ZeroRotator, VectorLocation, FVector(ParticleRenderRadius / ParticleMeshRadius)),
				true);
			//UE_LOG(LogTemp, Warning, TEXT("%u: Boundary Location=(%f, %f, %f)"), BoundaryParticleIndex, VectorLocation.X, VectorLocation.Y, VectorLocation.Z);
			//UE_LOG(LogTemp, Warning, TEXT("%u: Boundary Location=(%f, %f, %f)"), BoundaryParticleIndex, HostPositions[BoundaryParticleIndex * 4], HostPositions[BoundaryParticleIndex * 4 + 2], HostPositions[BoundaryParticleIndex * 4 + 1]);
		}
		cudaMemcpy(HostPositions, DevicePositions, sizeof(float) * 4u * NumFluidParticles, cudaMemcpyDeviceToHost);
#if RENDER_INSTANCES

		//UE_LOG(LogTemp, Warning, TEXT("Tick::NumParticles: %u"), NumParticles);
		for (uint32 FluidParticleIndex = 0; FluidParticleIndex < NumRenderingFluidParticles; ++FluidParticleIndex)
		{
			FVector VectorLocation(HostPositions[FluidParticleIndex * 4] * ParticleRenderRadius / ParticleRadius,
				HostPositions[FluidParticleIndex * 4 + 2] * ParticleRenderRadius / ParticleRadius,
				(HostPositions[FluidParticleIndex * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
			//	//Particles[i]->SetActorLocation(VectorLocation);
			ParticleInstancedMeshComponent->UpdateInstanceTransform(FluidParticleIndex,
				FTransform(FRotator::ZeroRotator, VectorLocation, FVector(ParticleRenderRadius / ParticleMeshRadius)),
				true);
			//UE_LOG(LogTemp, Warning, TEXT("%u: Location=(%f, %f, %f)"), Index, VectorLocation.X, VectorLocation.Y, VectorLocation.Z);
		}
		//for (uint32 Index = NumFluidParticles; Index < NumParticles; ++Index)
		//{
		//	//UE_LOG(LogTemp, Warning, TEXT("[%u]: Boundary Position=(%f, %f, %f)"), Index, HostPositions[Index * 4], HostPositions[Index * 4 + 1], HostPositions[Index * 4 + 2], HostPositions[Index * 4 + 3]);
		//	FVector VectorLocation(HostPositions[Index * 4] * ParticleRenderRadius / ParticleRadius,
		//		HostPositions[Index * 4 + 2] * ParticleRenderRadius / ParticleRadius,
		//		(HostPositions[Index * 4 + 1] + 1.0f) * ParticleRenderRadius / ParticleRadius);
		//	////	//Particles[i]->SetActorLocation(VectorLocation);
		//	//BoundaryParticleInstancedMeshComponent->UpdateInstanceTransform(Index - NumFluidParticles,
		//	//	FTransform(FRotator::ZeroRotator, VectorLocation, FVector(ParticleRenderRadius / ParticleMeshRadius)),
		//	//	true);
		//	//UE_LOG(LogTemp, Warning, TEXT("%u: Boundary Location=(%f, %f, %f)"), Index, VectorLocation.X, VectorLocation.Y, VectorLocation.Z);
		//}
#endif
		break;
	default:
		break;
	}

#if !RENDER_INSTANCES
	// marching cubes algorithm
	uint32 Threads = 128u;
	dim3 Grid(NumVoxels / Threads, 1u, 1u);

	// get around maximum Grid size of 65535 in each dimension
	if (Grid.x > 65535u)
	{
		Grid.y = Grid.x / 32768u;
		Grid.x = 32768u;
	}

#if SAMPLE_VOLUME
	static uint Size = McGridSize.x * McGridSize.y * McGridSize.z * sizeof(uchar);
	if (DeviceVolumes == nullptr)
	{
		cudaMalloc(reinterpret_cast<void**>(&DeviceVolumes), Size);
	}

	CudaCreateVolumeFromMassAndDensities(Grid,
		Threads,
		DeviceVolumes,
		McGridSize,
		GridSizeShift,
		GridSizeMask,
		VoxelSize,
		NumFluidParticles,
		reinterpret_cast<float4*>(DeviceMcSortedPositions),
		DeviceMcGridParticleIndices,
		DeviceMcCellStarts,
		DeviceMcCellEnds);
	CudaCreateVolumeTexture(DeviceVolumes, Size);
	uchar* HostVolumes = reinterpret_cast<uchar*>(FMemory::Malloc(Size));
	cudaMemcpy(HostVolumes, DeviceVolumes, Size, cudaMemcpyDeviceToHost);

	//for (uint32 i = 0u; i < Size; ++i)
	//{
	//	UE_LOG(LogTemp, Warning, TEXT("[%u]: volume: %u"), i, HostVolumes[i]);
	//}
#endif

	// calculate number of vertices need per voxel
	CudaLaunchClassifyVoxels(Grid, 
							 Threads,
							 DeviceVoxelVerticess, 
							 DeviceVoxelsOccupied, 
							 DeviceVolumes,
							 GridSize, 
							 GridSizeShift, 
							 GridSizeMask,
							 NumVoxels, 
							 VoxelSize, 
							 IsoValue, 
							 DeviceSortedPositions, 
							 DeviceGridParticleIndices, 
							 DeviceCellStarts, 
							 DeviceCellEnds,
							 NumFluidParticles,
							 NumRenderingFluidParticles);

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
		NumTotalVertices = 0;
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
	CudaThrustScanWrapper(DeviceVoxelVerticessScan, DeviceVoxelVerticess, NumVoxels);

#if DEBUG_BUFFERS
	printf("voxelVertsScan:\n");
	dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

	// readback total number of vertices
	{
		uint LastElement;
		uint LastScanElement;
		checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(&LastElement),
								   reinterpret_cast<void*>(DeviceVoxelVerticess + NumVoxels - 1),
								   sizeof(uint), 
								   cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(reinterpret_cast<void*>(&LastScanElement),
								   reinterpret_cast<void*>(DeviceVoxelVerticessScan + NumVoxels - 1),
								   sizeof(uint), 
								   cudaMemcpyDeviceToHost));
		NumTotalVertices = LastElement + LastScanElement;
		//UE_LOG(LogTemp, Warning, TEXT("Number of vertices after compacting: %u"), NumTotalVertices);
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
	//UE_LOG(LogTemp, Warning, TEXT("grid size: %u, %u, %u"), GridSize.x, GridSize.y, GridSize.z);
	//UE_LOG(LogTemp, Warning, TEXT("mc grid size: %u, %u, %u"), McGridSize.x, McGridSize.y, McGridSize.z);

#if SAMPLE_VOLUME
	CudaLaunchGenerateTriangles2(Grid2,
								 NTHREADS,
								 DeviceMcPositions, 
								 DeviceMcNormals,
								 DeviceCompactedVoxelArray,
								 DeviceVoxelVerticessScan, 
								 DeviceVolumes,
								 McGridSize,
								 GridSizeShift, 
								 GridSizeMask,
								 VoxelSize, 
								 IsoValue, 
								 NumActiveVoxels,
								 NumMaxVertices,
								 reinterpret_cast<float4*>(DeviceMcSortedPositions), 
								 DeviceMcCellStarts, 
								 DeviceMcCellEnds);
#else
	CudaLaunchGenerateTriangles(Grid2, 
								NTHREADS, 
								DeviceMcPositions, 
								DeviceMcNormals,
								DeviceCompactedVoxelArray,
								DeviceVoxelVerticessScan,
								GridSize, 
								GridSizeShift, 
								GridSizeMask,
								VoxelSize, 
								IsoValue, 
								NumActiveVoxels,
								NumMaxVertices, 
								DeviceSortedPositions, 
								DeviceGridParticleIndices, 
								DeviceCellStarts, 
								DeviceCellEnds, 
								NumFluidParticles,
								NumRenderingFluidParticles);
#endif
#endif

	BoundaryParticleInstancedMeshComponent->MarkRenderStateDirty();
#if RENDER_INSTANCES
	ParticleInstancedMeshComponent->MarkRenderStateDirty();
#else
	CreateIsosurface();
#endif
}

