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
#define GRID_SIZE (64u)
#define GRID_SIZE_LOG2  (6)
#define RENDER_INSTANCES (1)

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
		void InitializeGrid(int32 Size, float Spacing, float Jitter, int32 InNumParticles);

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

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		ESphPlatform SphPlatform = ESphPlatform::E_GPU_CUDA;

	int32 CudaSelectedDeviceId;
	GPU_INFO GpuInfo;
	bool bCanPrefetch;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ParticleMeshRadius = 50.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ParticleRenderRadius = 15.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float ParticleRadius = 1.0f / 64.0f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float BoundaryDamping = -0.5f;

	UPROPERTY(BlueprintReadWrite, EditDefaultsOnly)
		float Mass = 0.02f;

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

	// data
	uint32 NumParticles = NUM_PARTICLES;

	// CPU data
	float* HostPositions;
	float* HostVelocities;
	float* HostForces;
	float* HostPressureForces;
	float* HostViscosityForces;
	//float* HostSurfaceTensionForces;
	float* HostDensities;
	float* HostPressures;

	uint32* HostParticleHash;
	uint32* HostCellStarts;
	uint32* HostCellEnds;

	// GPU data
	float* DevicePositions;
	float* DeviceVelocities;
	float* DeviceForces;
	float* DevicePressureForces;
	float* DeviceViscosityForces;
	//float* DeviceSurfaceTensionForces;
	float* DeviceDensities;
	float* DevicePressures;

	float* DeviceSortedPositions;
	float* DeviceSortedVelocities;

	// grid data for sorting method
	uint32* DeviceGridParticleHashes;
	uint32* DeviceGridParticleIndice;
	uint32* DeviceCellStarts;
	uint32* DeviceCellEnds;

	uint32 GridSortBits;

	float* CudaPositionsVbo;	// these are the CUDA deviceMem Positions

	CudaSimParams SimulationParameters;
	uint3 GridSize;
	uint32 NumGridCells;

	uint32 SolverIterations;

	//UPROPERTY(BlueprintReadWrite, VisibleAnywhere)
	//TArray<class AParticlesActor*> Particles;

	// marching cubes
	uint3 GridSizeLog2;
	uint3 GridSizeMask;
	uint3 GridSizeShift;

	float3 VoxelSize;
	uint NumVoxels;
	uint NumMaxVertice;
	uint NumActiveVoxels;
	uint NumTotalVertice;

	float IsoValue;
	float DeviceIsoValue;

	// device data
	float4* DeviceMcPositions;
	float4* DeviceMcNormals;

	float4* HostMcVertices;
	float4* HostMcNormals;

	uchar* DeviceVolumes;
	uint32* DeviceVoxelVertices;
	uint32* DeviceVoxelVerticesScan;
	uint32* DeviceVoxelsOccupied;
	uint32* DeviceVoxelsOccupiedScan;
	uint32* DeviceCompactedVoxelArray;

	// tables
	uint32* DeviceNumVerticesTable;
	uint32* DeviceEdgeTable;
	uint32* DeviceTriTable;

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
		UProceduralMeshComponent* ParticleProceduralMeshComponent = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Components")
		class UMaterialInterface* Material = nullptr;
};
