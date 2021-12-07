// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#define WIN32_LEAN_AND_MEAN

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ProceduralMeshComponent.h"

#include <cuda_runtime.h>

#include <helper_functions.h> // helper utility functions 
#include <helper_cuda.h>      // helper functions for CUDA error checking and initialization

#include "helper_math.h"

#include "CUDA_Util.h"
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "defines.h"
#include "particleSystem.cuh"

#include "ParticlesActor.generated.h"

//#define NUM_PARTICLES (65536u)
//#define NUM_PARTICLES (32768u)
#define NUM_PARTICLES (16384u)
//#define NUM_PARTICLES (8192u)
//#define NUM_PARTICLES (4096u)
//#define NUM_PARTICLES (2048u)
//#define NUM_PARTICLES (1024u)
//#define NUM_PARTICLES (512u)
//#define NUM_PARTICLES (128u)
#define NUM_BOUNDARY_PARTICLES (4096u)
#define NUM_BOUNDARY_PARTICLES_PER_FISH (16u)
#define GRID_SIZE (64u)
#define GRID_SIZE_LOG2  (7u)
#define RENDER_INSTANCES (0)

UENUM(BlueprintType)
enum class ESphPlatform : uint8
{
	E_CPU_SINGLE_THREAD UMETA(DisplayName = "CPU Single Thread"),
	E_CPU_MULTIPLE_THREADS UMETA(DisplayName = "CPU Multiple Threads"),
	E_GPU_CUDA UMETA(DisplayName = "GPU CUDA")
};

UCLASS()
class PX_API AParticlesActor : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	AParticlesActor();
	virtual ~AParticlesActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	UFUNCTION(BlueprintCallable)
		void Initialize();

	UFUNCTION(BlueprintCallable)
		bool InitializeCuda();

	bool InitializeCuda(GPU_SELECT_MODE Mode, int32 SpecifiedDeviceId);

	UFUNCTION(BlueprintCallable)
		void InitializeGrid(int32 Size, float Spacing, float Jitter, int32 InNumFluidParticles, int32 InBoundaryParticles);

	UFUNCTION(BlueprintCallable)
		void CleanUpCuda();

	UFUNCTION(BlueprintCallable)
		void Destroy();

	uint32 CalculateGridHash(int3 GridPosition);
	int3 CalculateGridPosition(float3 Position);
	void CalculateHash(uint32* GridParticleHashes, uint32* GridParticleIndices, float* Positions);

	UFUNCTION(BlueprintCallable)
		float CalculatePoly6BySquaredDistance(float SquaredDistance);

	UFUNCTION(BlueprintCallable)
		FVector CalculatePoly6Gradient(const FVector& Vector);

	float3 CalculatePoly6Gradient(float3 Vector);

	UFUNCTION(BlueprintCallable)
		FVector CalculateSpikyGradient(const FVector& Vector);

	float3 CalculateSpikyGradient(float3 Vector);

	UFUNCTION(BlueprintCallable)
		float CalculateViscosityLaplacianByDistance(float Distance);

	void ComputeDensityAndPressure(float* OutDensities, float* OutPressures, float* SortedPositions, uint32* GridParticleIndices, uint32* CellStart, uint32* CellEnd, uint32 InNumParticles, uint32 InNumCells);

	void ComputeVelocities(float* OutVelocities, float* SortedPositions, float* SortedVelocities, float* Densities, float* Pressures, float DeltaTime, uint32* GridParticleIndices, uint32* CellStart, uint32* CellEnd, uint32 InNumParticles, uint32 NumCells);

	void Integrate(float* Positions, float* Velocities, float DeltaTime, uint32 InNumParticles);

	void SortParticles(uint32* GridParticleHashes, uint32* GridParticleIndices);

	UFUNCTION(BlueprintCallable)
		void Reset();

	UFUNCTION(BlueprintCallable)
		void CreateIsosurface();

	UFUNCTION(BlueprintImplementableEvent)
		void UpdateFishesLocation();

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		ESphPlatform SphPlatform = ESphPlatform::E_GPU_CUDA;

	int32 CudaSelectedDeviceId;
	GPU_INFO GpuInfo;
	bool bCanPrefetch;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ParticleMeshRadius = 50.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ParticleRenderRadius = 10.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ParticleRadius = 1.0f / 64.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float BoundaryDamping = -0.5f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float FluidParticleMass = 0.02f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float BoundaryParticleMass = 0.05f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float SupportRadius;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float SupportRadiusSquared;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float StiffnessConstant = 3.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float RestDensity = 998.29f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float Viscosity = 0.01f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float Threshold = 7.065f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ThresholdSquared;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float SurfaceTension = 0.0728f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float CflScale = 0.4f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float Poly6;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float Poly6Gradient;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float Poly6Laplacian;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float SpikyGradient;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ViscosityLaplacian;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		FVector Gravity = FVector(0.0f, 0.0f, -9.80665f);

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float CustomDeltaTime = 0.001f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		int32 NumFishes;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		TArray<FVector> FishesCoordinate;

	// data
	uint32 NumParticles = NUM_PARTICLES + NUM_BOUNDARY_PARTICLES;
	uint32 NumFluidParticles = NUM_PARTICLES;
	uint32 NumBoundaryParticles = NUM_BOUNDARY_PARTICLES;

	// CPU data
	float* HostPositions = nullptr;
	float* HostVelocities = nullptr;
	float* HostForces = nullptr;
	float* HostPressureForces = nullptr;
	float* HostViscosityForces = nullptr;
	//float* HostSurfaceTensionForces = nullptr;
	float* HostDensities = nullptr;
	float* HostPressures = nullptr;

	uint32* HostGridParticleHashes = nullptr;
	uint32* HostGridParticleIndice = nullptr;
	uint32* HostCellStarts = nullptr;
	uint32* HostCellEnds = nullptr;

	// GPU data
	float* DevicePositions = nullptr;
	float* DeviceVelocities = nullptr;
	float* DeviceForces = nullptr;
	float* DevicePressureForces = nullptr;
	float* DeviceViscosityForces = nullptr;
	//float* DeviceSurfaceTensionForces = nullptr;
	float* DeviceDensities = nullptr;
	float* DevicePressures = nullptr;

	float* DeviceSortedPositions = nullptr;
	float* DeviceSortedVelocities = nullptr;

	// grid data for sorting method
	uint32* DeviceGridParticleHashes = nullptr;
	uint32* DeviceGridParticleIndice = nullptr;
	uint32* DeviceCellStarts = nullptr;
	uint32* DeviceCellEnds = nullptr;

	uint32 GridSortBits;

	float* CudaPositionsVbo = nullptr;	// these are the CUDA deviceMem Positions

	CudaSimParams SimulationParameters;
	uint3 GridSize;
	uint32 NumGridCells;

	uint32 SolverIterations = 1u;

	//UPROPERTY(BlueprintReadWrite, VisibleAnywhere)
	//TArray<class AParticlesActor*> Particles;

	// marching cubes
	uint3 GridSizeLog2;
	uint3 McGridSize;
	uint3 GridSizeMask;
	uint3 GridSizeShift;

	float3 VoxelSize;
	uint NumVoxels;
	uint NumMaxVertice;
	uint NumActiveVoxels = 0u;
	uint NumTotalVertice = 0u;

	float IsoValue;
	float DeviceIsoValue;

	float4* HostMcVertices = nullptr;
	float4* HostMcNormals = nullptr;

	uint32* HostMcCellStarts = nullptr;
	uint32* HostMcCellEnds = nullptr;

	float* DeviceMcSortedPositions = nullptr;

	// grid data for sorting method
	uint32* DeviceMcGridParticleHashes = nullptr;
	uint32* DeviceMcGridParticleIndice = nullptr;
	uint32* DeviceMcCellStarts = nullptr;
	uint32* DeviceMcCellEnds = nullptr;

	// device data
	float4* DeviceMcPositions = nullptr;
	float4* DeviceMcNormals = nullptr;

	uchar* DeviceVolumes = nullptr;
	uint32* DeviceVoxelVertices = nullptr;
	uint32* DeviceVoxelVerticesScan = nullptr;
	uint32* DeviceVoxelsOccupied = nullptr;
	uint32* DeviceVoxelsOccupiedScan = nullptr;
	uint32* DeviceCompactedVoxelArray = nullptr;

	// tables
	uint32* DeviceNumVerticesTable = nullptr;
	uint32* DeviceEdgeTable = nullptr;
	uint32* DeviceTriTable = nullptr;

	TArray<FVector> Vertices;
	TArray<int32> Triangles;
	TArray<FVector> Normals;
	TArray<FVector2D> Uv0;
	TArray<FColor> VertexColors;
	TArray<FProcMeshTangent> Tangents;

	// Particles
	//UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UInstancedStaticMeshComponent* ParticleInstancedMeshComponent = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UInstancedStaticMeshComponent* BoundaryParticleInstancedMeshComponent = nullptr;
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		UProceduralMeshComponent* ParticleProceduralMeshComponent = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UMaterialInterface* Material = nullptr;
};
