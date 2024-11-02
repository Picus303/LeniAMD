#include "htc/kernels.hpp"

#include "lve/utils.hpp"

// Lenia parameters
__constant__ float MU = 20.0f;
__constant__ float SIGMA = 5.0f;
__constant__ float ALPHA = 0.1f;


// This kernel updates the state of the simulation based on the results of the convolution
__global__ void updateKernel(int width, int height, int depth, float* state, float* intermediate) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x < width && y < height && z < depth) {
		int idx = z * width * height + y * width + x;

		float normalized = (intermediate[idx] - MU) / SIGMA;
		float t = expf(-normalized * normalized);

		state[idx] = (1 - ALPHA) * state[idx] + ALPHA * t;
	}
}

// This kernel colors the vertices based on the state of the simulation
__global__ void colorKernel(int width, int height, float* state, lve::Vertex* outputVertexArray) {
	__shared__ lve::Vertex sharedOutput[BLOCK_SIZE_X * BLOCK_SIZE_Y];
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int localIdx = threadIdx.y * blockDim.x + threadIdx.x;

	if (x < width && y < height) {
		int idx_r = 0 * width * height + y * width + x;
		int idx_g = 1 * width * height + y * width + x;
		int idx_b = 2 * width * height + y * width + x;

		int globalIdx = y * width + x;

		sharedOutput[localIdx].position.x = (2.0f * x / width - 1.0f);
		sharedOutput[localIdx].position.y = (2.0f * y / height - 1.0f);

		sharedOutput[localIdx].color.r = state[idx_r];
		sharedOutput[localIdx].color.g = state[idx_g];
		sharedOutput[localIdx].color.b = state[idx_b];

		outputVertexArray[globalIdx] = sharedOutput[localIdx];
	}
}