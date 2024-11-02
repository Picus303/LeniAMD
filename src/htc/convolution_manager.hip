#include "htc/convolution_manager.hpp"
#include "htc/utils.hpp"

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>


#define KERNEL_SIZE 31


namespace htc {

	ConvolutionManager::ConvolutionManager(int width, int height, int depth, float* input, float* output) :
		width(width), height(height), depth(depth), input(input), output(output) {

		// Create the MIOpen context
		CHECK_HIP_ERROR(hipStreamCreate(&stream));
		CHECK_MIOPEN_ERROR(miopenCreateWithStream(&handle, stream));

		// Set the convolution parameters
		init_kernels();
		set_descriptors();
		find_algorithm();
	}

	ConvolutionManager::~ConvolutionManager() {
		// Ensure that computations are finished
		CHECK_HIP_ERROR(hipStreamSynchronize(stream));		

		// Release resources
		CHECK_MIOPEN_ERROR(miopenDestroyTensorDescriptor(inputDescriptor));
		CHECK_MIOPEN_ERROR(miopenDestroyTensorDescriptor(outputDescriptor));
		CHECK_MIOPEN_ERROR(miopenDestroyTensorDescriptor(kernelDescriptor));
		CHECK_MIOPEN_ERROR(miopenDestroyConvolutionDescriptor(convolutionDescriptor));
		CHECK_MIOPEN_ERROR(miopenDestroy(handle));
		CHECK_HIP_ERROR(hipStreamDestroy(stream));
		CHECK_HIP_ERROR(hipFree(kernel));
		if (workspaceSize > 0) {
			CHECK_HIP_ERROR(hipFree(workspace));
		}
	}

	void ConvolutionManager::init_kernels() {
		// Allocate required memory
		float* h_kernel = new float[depth * depth * KERNEL_SIZE * KERNEL_SIZE];
		CHECK_HIP_ERROR(hipMalloc(&kernel, depth * depth * KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
		
		// Initialize kernel weights
		for (int i = 0; i < depth; i++) {
			fill_guassian_kernel(i, (i + 0) % depth, 4.0, 1.0, h_kernel);
			fill_guassian_kernel(i, (i + 1) % depth, 8.0, 2.0, h_kernel);
			fill_guassian_kernel(i, (i + 2) % depth, 12.0, 3.0, h_kernel);
		}

		// Copy kernel weights to the GPU
		CHECK_HIP_ERROR(hipMemcpy(kernel, h_kernel, depth * depth * KERNEL_SIZE * KERNEL_SIZE * sizeof(float), hipMemcpyHostToDevice));
		delete[] h_kernel;
	}

	void ConvolutionManager::fill_guassian_kernel(int sourceChannel, int targetChannel, float mu, float sigma, float* h_kernel) {
		int globalIdx = targetChannel * (depth * KERNEL_SIZE * KERNEL_SIZE)	// F: Position in the target channel
		  + sourceChannel * (KERNEL_SIZE * KERNEL_SIZE);        			// C: Position in the source channel

		int locX, locY, localIdx;
		float distDelta, normalized;

		// Traverse the kernel centering the values
		int halfSize = (KERNEL_SIZE - 1) / 2;
		for (int h = -halfSize; h <= halfSize; h++) {
			for (int w = -halfSize; w <= halfSize; w++) {
				// Calculate local position
				locX = w + halfSize;
				locY = h + halfSize;
				localIdx = locY * KERNEL_SIZE + locX;

				// Calculate normalized distance to the center
				distDelta = sqrtf(h * h + w * w) - mu;
				normalized = distDelta * distDelta / (2 * sigma * sigma);
				h_kernel[globalIdx + localIdx] = expf(-normalized);
			}
		}
	}

	void ConvolutionManager::set_descriptors() {
		int pad = (KERNEL_SIZE - 1) / 2;
		int stride = 1;
		int dilation = 1;

		// Create descriptors
		CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&inputDescriptor));
		CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&outputDescriptor));
		CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&kernelDescriptor));
		CHECK_MIOPEN_ERROR(miopenCreateConvolutionDescriptor(&convolutionDescriptor));

		// Set descriptors
		CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(inputDescriptor, miopenFloat, 1, depth, height, width));
		CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(outputDescriptor, miopenFloat, 1, depth, height, width));
		CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(kernelDescriptor, miopenFloat, depth, depth, KERNEL_SIZE, KERNEL_SIZE));
		CHECK_MIOPEN_ERROR(miopenInitConvolutionDescriptor(convolutionDescriptor, miopenConvolution, pad, pad, stride, stride, dilation, dilation));
	}

	void ConvolutionManager::find_algorithm() {
		// Find the convolution algorithm
		int returnedAlgoCount;
		miopenConvAlgoPerf_t perfResults;

		CHECK_MIOPEN_ERROR(miopenFindConvolutionForwardAlgorithm(handle,
																inputDescriptor, input,
																kernelDescriptor, kernel,
																convolutionDescriptor,
																outputDescriptor, output,
																1, &returnedAlgoCount, &perfResults,
																workspace, workspaceSize, false));

		// Check if an algorithm was found
		if (returnedAlgoCount == 0) {
			throw std::runtime_error("No convolution algorithm found");
		}

		// Select the algorithm
		convolutionAlgorithm = perfResults.fwd_algo;

		// Allocate memory for the workspace
		if (workspaceSize > 0) {
			CHECK_HIP_ERROR(hipMalloc(&workspace, workspaceSize));
		}
	}

	void ConvolutionManager::runConvolution() {
		// Run the convolution and wait for it to finish
		CHECK_MIOPEN_ERROR(miopenConvolutionForward(handle,
													&alpha, inputDescriptor, input,
													kernelDescriptor, kernel,
													convolutionDescriptor, convolutionAlgorithm,
													&beta, outputDescriptor, output,
													workspace, workspaceSize));

		CHECK_HIP_ERROR(hipStreamSynchronize(stream));
	}
}