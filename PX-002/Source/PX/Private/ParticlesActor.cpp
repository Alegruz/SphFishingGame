// Fill out your copyright notice in the Description page of Project Settings.


#include "ParticlesActor.h"

#include <cuda_runtime.h>

#include "helper_cuda.h"

#include "particleSystem.cuh"

// Sets default values
AParticlesActor::AParticlesActor()
	: m_bInitialized(false)
	, m_hPos(nullptr)
	, m_hVel(nullptr)
	, m_dPos(nullptr)
	, m_dVel(nullptr)
	, m_timer(nullptr)
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

    InstancedStaticMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("RootComponent"));
    RootComponent = InstancedStaticMeshComponent;

	NumParticles = NUM_PARTICLES;
	uint gridDim = GRID_SIZE;
	GridSize.x = GridSize.y = GridSize.z = gridDim;

	char* ArgumentVector[] = { "KKE's SPH" };
    //¼Ò¸êÀÚ¿¡´Â Clear Instance
	cudaInit(1, ArgumentVector);

	cudaMallocHost(reinterpret_cast<void**>(&HostPositions), sizeof(float) * 3u);
	cudaMalloc(reinterpret_cast<void**>(&DevicePositions), sizeof(float) * 3u);

	m_numGridCells = GridSize.x * GridSize.y * GridSize.z;
	//float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 18;    // increase this for larger grids

	// set simulation parameters
	m_params.gridSize = GridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = NumParticles;

	m_params.particleRadius = 1.0f / 64.0f;
	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = 0.2f;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	//m_params.cellSize = make_float3(worldSize.x / GridSize.x, worldSize.y / GridSize.y, worldSize.z / GridSize.z);
	float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.0f;
	m_params.boundaryDamping = -0.5f;

	m_params.particleMass = 0.02f;

	m_params.gravity = make_float3(0.0f, -9.82f, 0.0f);
	m_params.globalDamping = 1.0f;

	m_params.supportRadius = 0.0457f;

}

AParticlesActor::~AParticlesActor()
{
	Destroy();
	NumParticles = 0u;
}

// Called when the game starts or when spawned
void AParticlesActor::BeginPlay()
{
	Super::BeginPlay();
    Initialize(NumParticles);
    Reset();
    //initPrams();
}

// Called every frame
void AParticlesActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
    Update(timestep);
}



void AParticlesActor::Initialize(uint32 numParticles)
{
    assert(!m_bInitialized);

	NumParticles = numParticles;

    m_hPos = new float[NumParticles * 4];
    m_hVel = new float[NumParticles * 4];
    m_hForces = new float[NumParticles * 4];
    m_hDensities = new float[NumParticles];
    m_hPressures = new float[NumParticles];
    memset(m_hPos, 0, NumParticles * 4 * sizeof(float));
    memset(m_hVel, 0, NumParticles * 4 * sizeof(float));
    memset(m_hForces, 0, NumParticles * 4 * sizeof(float));
    memset(m_hDensities, 0, NumParticles * sizeof(float));
    memset(m_hPressures, 0, NumParticles * sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells *  sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * NumParticles;

    checkCudaErrors(cudaMalloc((void**)&m_cudaPosVBO, memSize));

    allocateArray((void**)&m_dVel, memSize);
    allocateArray((void**)&m_dForces, memSize);
    allocateArray((void**)&m_dDensities, memSize);
    allocateArray((void**)&m_dPressures, memSize);

    allocateArray((void**)&m_dSortedPos, memSize);
    allocateArray((void**)&m_dSortedVel, memSize);

    allocateArray((void**)&m_dGridParticleHash, NumParticles * sizeof(uint));
    allocateArray((void**)&m_dGridParticleIndex, NumParticles * sizeof(uint));

    allocateArray((void**)&m_dCellStart, m_numGridCells * sizeof(uint));
    allocateArray((void**)&m_dCellEnd, m_numGridCells * sizeof(uint));

	checkCudaErrors(cudaMalloc((void**)&m_cudaColorVBO, sizeof(float) * numParticles * 4));

    // add instances
    for (uint32 i = 0; i < NumParticles; i++)
    {
        InstancedStaticMeshComponent->AddInstance(FTransform());
    }

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void AParticlesActor::Destroy()
{
    assert(m_bInitialized);

    delete[] m_hPos;
    delete[] m_hVel;
    delete[] m_hForces;
    delete[] m_hDensities;
    delete[] m_hPressures;

    delete[] m_hCellStart;
    delete[] m_hCellEnd;

    freeArray(m_dVel);
    freeArray(m_dForces);
    freeArray(m_dDensities);
    freeArray(m_dPressures);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    checkCudaErrors(cudaFree(m_cudaPosVBO));
    checkCudaErrors(cudaFree(m_cudaColorVBO));
}

void AParticlesActor::Reset()
{
    float jitter = m_params.particleRadius * 0.01f;
    uint s = (int)ceilf(powf((float)NumParticles, 1.0f / 3.0f));
    uint gridSize[3];
    gridSize[0] = gridSize[1] = gridSize[2] = s;
    InitGrid(gridSize, m_params.particleRadius * 2.0f, jitter, NumParticles);

    SetArray(POSITION, m_hPos, 0, NumParticles);
    SetArray(VELOCITY, m_hVel, 0, NumParticles);
    SetArray(FORCE, m_hForces, 0, NumParticles);
    SetArray(DENSITY, m_hDensities, 0, NumParticles);
    SetArray(PRESSURE, m_hPressures, 0, NumParticles);
}

inline float frand()
{
    return rand() / (float)RAND_MAX;
}

//Using in Reset()
void AParticlesActor::InitGrid(uint32* size, float spacing, float jitter, uint32 numParticles)
{
    srand(1973);

    for (uint z = 0; z < size[2]; z++)
    {
        for (uint y = 0; y < size[1]; y++)
        {
            for (uint x = 0; x < size[0]; x++)
            {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i * 4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 3] = 1.0f;

                    m_hVel[i * 4] = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;

                    // set position of instances
                    bool bResult = InstancedStaticMeshComponent->UpdateInstanceTransform(i, FTransform(FRotator::ZeroRotator,
                        FVector(m_hPos[i * 4] * 10.f * 64.f, m_hPos[i * 4 + 2] * 10.f * 64.f, (m_hPos[i * 4 + 1] + 1.0f) * 10.f * 64.f),
                        FVector(10.f / 50.f, 10.f / 50.f, 10.f / 50.f)));
                    //UE_LOG(LogTemp, Warning, TEXT("position: %f, %f, %f"), m_hPos[i * 4] * 100, m_hPos[i * 4 + 2] * 100, m_hPos[i * 4 + 1] * 100);

                    if (!bResult)
                    {
                        UE_LOG(LogTemp, Warning, TEXT("[%u] results false"), i);
                    }

                    InstancedStaticMeshComponent->MarkRenderStateDirty();
                }
            }
        }
    }
    //InstancedStaticMeshComponent->MarkRenderStateDirty();
}

//Using in Reset()
void AParticlesActor::SetArray(ParticleArray array, const float* data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
        {
            copyArrayToDevice(m_cudaPosVBO, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        }
        break;
        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
            break;
        case FORCE:
            copyArrayToDevice(m_dForces, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
            break;
        case DENSITY:
            copyArrayToDevice(m_dDensities, data, start * sizeof(float), count * sizeof(float));
            break;
        case PRESSURE:
            copyArrayToDevice(m_dPressures, data, start * sizeof(float), count * sizeof(float));
            break;
    }
}

/*
void AParticlesActor::initPrams()
{
    params = new ParamListGL("misc");
    params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
    params->AddParam(new Param<float>("damping", damping, 0.0f, 1.0f, 0.001f, &damping));
    params->AddParam(new Param<float>("gravity", gravity, 0.0f, 0.001f, 0.0001f, &gravity));
    params->AddParam(new Param<int>("ball radius", ballr, 1, 20, 1, &ballr));

    params->AddParam(new Param<float>("collide spring", collideSpring, 0.0f, 1.0f, 0.001f, &collideSpring));
    params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
    params->AddParam(new Param<float>("collide shear", collideShear, 0.0f, 0.1f, 0.001f, &collideShear));
    params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));
}
*/

void AParticlesActor::Update(float deltaTime)
{
    assert(m_bInitialized);

    float* dPos;
    dPos = (float*)m_cudaPosVBO;

    // update constants
    setParameters(&m_params);

    //def computeDensity
    //computeDensity(dPos);

    //def computePressure
    //def computeForse

    // integrate
    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        NumParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, NumParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        dPos,
        m_dVel,
        NumParticles,
        m_numGridCells);

    // process collisions
    computeDensityAndPressure(
        m_dDensities,
        m_dPressures,
        m_dSortedPos,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        NumParticles,
        m_numGridCells);

    computeForces(m_dVel, m_dForces,
        deltaTime,
        m_dSortedPos,
        m_dSortedVel,
        m_dDensities,
        m_dPressures,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        NumParticles,
        m_numGridCells);

    integrateSystem(
        dPos,
        m_dVel,
        deltaTime,
        NumParticles);

    //cudaMemcpy(m_hVel, m_dVel, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
    //UE_LOG(LogTemp, Warning, TEXT("Pos: %f, %f, %f"), m_hPos);

    // copy device data to host data
    cudaMemcpy(m_hPos, dPos, sizeof(float) * 4 * NumParticles, cudaMemcpyDeviceToHost);
    //UE_LOG(LogTemp, Warning, TEXT("Pos: %f, %f, %f"), m_hPos);

    // update positions
    for (uint32 i = 0; i < NumParticles; i++)
    {
        //if (FMath::IsNaN(m_hVel[4 * i]) || FMath::IsNaN(m_hVel[4 * i + 1]) || FMath::IsNaN(m_hVel[4 * i + 2]))
        //if (FMath::IsNaN(m_hForces[4 * i]) || FMath::IsNaN(m_hForces[4 * i + 1]) || FMath::IsNaN(m_hForces[4 * i + 2]))
        //{
        //    UE_LOG(LogTemp, Warning, TEXT("NAAN"));
        //}
        bool bResult = InstancedStaticMeshComponent->UpdateInstanceTransform(i, FTransform(FRotator::ZeroRotator,
            FVector(m_hPos[i * 4] * 10.f * 64.f, m_hPos[i * 4 + 2] * 10.f * 64.f, (m_hPos[i * 4 + 1] + 1.0f) * 10.f * 64.f),
            FVector(10.f / 50.f, 10.f / 50.f, 10.f / 50.f)));
        //UE_LOG(LogTemp, Warning, TEXT("position: %f, %f, %f"), m_hPos[i * 4] * 100, m_hPos[i * 4 + 2] * 100, m_hPos[i * 4 + 1] * 100);

        //if (!bResult)
        //{
        //    UE_LOG(LogTemp, Warning, TEXT("[%u] results false"), i);
        //}
    }
    //UE_LOG(LogTemp, Warning, TEXT("Render Num: %u"), InstancedStaticMeshComponent->GetNumRenderInstances());

    // mark render state dirty
    InstancedStaticMeshComponent->MarkRenderStateDirty();
}