#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <lve/utils.hpp>

#include <hip/hip_runtime.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

// NOTE: Each block contains 1024 threads, which might be too many for some GPUs
// NOTE: Might need to replace this with a dynamic approch

__global__ void updateKernel(int width, int height, int depth, float* state, float* intermediate);
__global__ void colorKernel(int width, int height, float* state, lve::Vertex* outputVertexArray);

#endif