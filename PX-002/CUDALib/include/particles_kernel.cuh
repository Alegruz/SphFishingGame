/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct CudaSimParams
{
    float3 Gravity;
    float ParticleRadius;

    float SupportRadius;
    float SupportRadiusSquared;

    uint3 GridSize;
    uint3 McGridSize;
    uint NumCells;
    uint NumVoxels;
    float3 WorldOrigin;
    float3 CellSize;
    float3 McCellSize;

    uint NumBodies;
    uint MaxParticlesPerCell;

    float BoundaryDamping;

    float ParticleMass;
    float BoundaryParticleMass;
    float GasConstant;
    float RestDensity;
    float Viscosity;
    float Threshold;
    float ThresholdSquared;
    float SurfaceTension;

    float CflScale;

    float Poly6;
    float Poly6Gradient;
    float Poly6Laplacian;
    float SpikyGradient;
    float ViscosityLaplacian;

    float DeltaTime;

    float XScaleFactor;
    float YScaleFactor;
    float ZScaleFactor;
};

#endif
