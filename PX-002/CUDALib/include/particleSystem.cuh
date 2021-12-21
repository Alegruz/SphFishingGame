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

extern "C"
{
    void CudaInit(int Argc, char** Argv);

    void CudaAllocateArray(void** DevicePtr, size_t Size);
    void CudaFreeArray(void* DevicePtr);

    void CudaThreadSync();

    void CudaCopyArrayToDevice(void* DeviceArray, const void* HostArray, size_t Offset, size_t Size);

    void CudaSetParameters(CudaSimParams* HostParameters);

    //cuda header

    void CudaIntegrateSystem(float* Positions,
                             float* Velocities,
                             uint NumParticles);

    void CudaCalculateHashes(uint*  OutGridParticleHashes,
                             uint*  OutGridParticleIndices,
                             float* Positions,
                             uint   NumParticles);

    void CudaMcCalculateHashes(uint*  OutGridParticleHashes,
                               uint*  OutGridParticleIndices,
                               float* Positions,
                               uint   NumParticles);

    void CudaReorderDataAndFindCellStart(uint* OutCellStarts,
                                         uint* OutCellEnds,
                                         float* OutSortedPositions,
                                         float* OutSortedVelocities,
                                         uint* GridParticleHashes,
                                         uint* GridParticleIndices,
                                         float* Positions,
                                         float* Velocities,
                                         uint   NumParticles,
                                         uint   NumCells);

    void CudaMcReorderDataAndFindCellStart(uint* OutCellStarts,
                                           uint* OutCellEnds,
                                           float* OutSortedPositions,
                                           uint* GridParticleHashes,
                                           uint* GridParticleIndices,
                                           float* Positions,
                                           uint   NumParticles,
                                           uint   NumVoxels);

    void CudaComputeDensitiesAndPressures(float* OutDensities,
                                          float* OutPressures,
                                          float* SortedPositions,
                                          uint*  GridParticleIndices,
                                          uint*  CellStarts,
                                          uint*  CellEnds,
                                          uint   NumBoundaryParticles,
                                          uint   NumFluidParticles,
                                          uint   NumRenderingFluidParticles,
                                          uint   NumParticles);

    void CudaComputeAllForcesAndVelocities(float* OutVelocities,
                                           float* OutForces,
                                           float* OutPressureForces,
                                           float* OutViscosityForces,
                                           float* SortedPositions, // input: sorted positions
                                           float* SortedVelocities,
                                           float* Densities,
                                           float* Pressures,
                                           uint*  GridParticleIndices,    // input: sorted particle indices
                                           uint*  CellStarts,
                                           uint*  CellEnds,
                                           uint   NumFluidParticles,
                                           uint   NumRenderingFluidParticles,
                                           uint   NumParticles);

    void CudaComputeForcesAndVelocities(float* OutVelocities,
                                        float* OutForces,
                                        float* SortedPositions, // input: sorted positions
                                        float* SortedVelocities,
                                        float* Densities,
                                        float* Pressures,
                                        uint*  GridParticleIndices,    // input: sorted particle indices
                                        uint*  CellStarts,
                                        uint*  CellEnds,
                                        uint   NumFluidParticles,
                                        uint   NumRenderingFluidParticles,
                                        uint   NumParticles);

    void CudaSortParticles(uint* DeviceGridParticleHashes, uint* DeviceGridParticleIndices, uint NumParticles);

    // marching cubes
    void CudaAllocateTextures(uint** DeviceEdgeTable, uint** DeviceTriTable, uint** DeviceNumVerticesTable);
    void CudaCreateVolumeTexture(uchar* DeviceVolumes, size_t BufferSize);
    void CudaDestroyAllTextureObjects();
    void CudaLaunchClassifyVoxels(dim3 Grid, 
                                  dim3 Threads, 
                                  uint* OutVoxelVertices, 
                                  uint* OutOccupiedVoxels, 
                                  uchar* Volumes,
                                  uint3 GridSize, 
                                  uint3 GridSizeShift, 
                                  uint3 GridSizeMask, 
                                  uint NumVoxels,
                                  float3 VoxelSize, 
                                  float IsoValue, 
                                  float* SortedPositions, 
                                  uint* GridParticleIndices, 
                                  uint* CellStarts, 
                                  uint* CellEnds,
                                  uint NumFluidParticles,
                                  uint NumRenderingFluidParticles);
    void CudaLaunchCompactVoxels(dim3 Grid, dim3 Threads, uint* OutCompactedVoxelArray, uint* OccupiedVoxels, uint* OccupiedScanVoxels, uint NumVoxels);
    void CudaLaunchGenerateTriangles(dim3 Grid, 
                                     dim3 Threads,
                                     float4* OutPositions, 
                                     float4* OutNormals, 
                                     uint* CompactedVoxelArray, 
                                     uint* NumScannedVertices,
                                     uint3 GridSize, 
                                     uint3 GridSizeShift, 
                                     uint3 GridSizeMask,
                                     float3 VoxelSize, 
                                     float IsoValue, 
                                     uint NumActiveVoxels, 
                                     uint NumMaxVertices, 
                                     float* SortedPositions,
                                     uint* GridParticleIndices,
                                     uint* CellStarts, 
                                     uint* CellEnds,
                                     uint NumFluidParticles,
                                     uint NumRenderingFluidParticles);
    void CudaLaunchGenerateTriangles2(dim3 Grid, 
                                      dim3 Threads,
                                      float4* OutPositions, 
                                      float4* OutNormals, 
                                      uint* CompactedVoxelArray, 
                                      uint* NumScannedVertices, 
                                      uchar* Volumes,
                                      uint3 GridSize, 
                                      uint3 GridSizeShift, 
                                      uint3 GridSizeMask,
                                      float3 VoxelSize, 
                                      float IsoValue, 
                                      uint NumActiveVoxels, 
                                      uint NumMaxVertices,
                                      float4* SortedPositions, 
                                      uint* GridParticleIndices,
                                      uint* CellStarts,
                                      uint* CellEnds, 
                                      uint NumFluidParticles,
                                      uint NumRenderingFluidParticles);
    void CudaThrustScanWrapper(unsigned int* Outputs, unsigned int* Inputs, unsigned int NumElements);

    void CudaCreateVolumeFromMassAndDensities(dim3 Grid, 
                                              dim3 Threads, 
                                              uchar* OutVolumes, 
                                              uint3 GridSize, 
                                              uint3 GridSizeShift, 
                                              uint3 GridSizeMask, 
                                              float3 VoxelSize, 
                                              uint NumFluidParticles,
                                              float4* SortedPositions,
                                              uint* GridParticleIndices,
                                              uint* CellStarts,
                                              uint* CellEnds);
}
