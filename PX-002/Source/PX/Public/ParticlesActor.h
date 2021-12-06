// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "ParticlesActor.generated.h"

#define GRID_SIZE       (64U)
//#define NUM_PARTICLES   (65536u)
//#define NUM_PARTICLES   (32768U)
//#define NUM_PARTICLES     (16384u)
#define NUM_PARTICLES   (1024u)

UCLASS()
class PX_API AParticlesActor : public AActor
{
	GENERATED_BODY()
	
public:	
    enum ParticleArray
    {
        POSITION,
        VELOCITY,
        FORCE,
        DENSITY,
        PRESSURE,
    };
	// Sets default values for this actor's properties
	AParticlesActor();
	virtual ~AParticlesActor();
    void Initialize(uint32 numParticles);
    void Destroy();
    void Reset();
    void InitGrid(uint32* size, float spacing, float jitter, uint32 numParticles);
    void SetArray(ParticleArray array, const float* data, int start, int count);
    //void initPrams();

    void Update(float deltaTime);

protected:
	// Called when the game starts or when    
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
    
protected:
    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Component")
    UInstancedStaticMeshComponent* InstancedStaticMeshComponent;

private:

	uint32 NumParticles = 0u; 
	uint3 GridSize;

	float* HostPositions = nullptr;
	float* DevicePositions = nullptr;

    bool m_bInitialized;

    // CPU data
    float* m_hPos;              // particle positions
    float* m_hVel;              // particle velocities
    float* m_hForces;
    float* m_hDensities;
    float* m_hPressures;

    uint32* m_hParticleHash;
    uint32* m_hCellStart;
    uint32* m_hCellEnd;

    // GPU data
    float* m_dPos;
    float* m_dVel;
    float* m_dForces;
    float* m_dDensities;
    float* m_dPressures;

    float* m_dSortedPos;
    float* m_dSortedVel;

    // grid data for sorting method
    uint32* m_dGridParticleHash; // grid hash value for each particle
    uint32* m_dGridParticleIndex;// particle index for each particle
    uint32* m_dCellStart;        // index of start of each cell in sorted list
    uint32* m_dCellEnd;          // index of end of cell

    uint32   m_gridSortBits;

    float* m_cudaPosVBO;        // these are the CUDA deviceMem Pos
    float* m_cudaColorVBO;      // these are the CUDA deviceMem Color

    struct cudaGraphicsResource* m_cuda_posvbo_resource; // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource* m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint32 m_numGridCells;

    StopWatchInterface* m_timer;

    float timestep = 0.007f;

    float ScaleRate = 64.0f / 640;
};
