#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>


namespace htc {

	// This class is responsible for managing all the ressources associated with MIOPEN and the convolution
	class ConvolutionManager {

		public:

			ConvolutionManager(int width, int height, int depth, float* input, float* output);
			~ConvolutionManager();

			// Not copyable or movable
			ConvolutionManager(const ConvolutionManager&) = delete;
			ConvolutionManager& operator=(const ConvolutionManager&) = delete;

			void runConvolution();

		private:

			hipStream_t stream;

			miopenHandle_t handle;

			// Descriptors
			miopenTensorDescriptor_t inputDescriptor;
			miopenTensorDescriptor_t outputDescriptor;
			miopenTensorDescriptor_t kernelDescriptor;

			// Convolution parameters
			const float alpha = 1.0f;
			const float beta = 0.0f;

			int width;
			int height;
			int depth;

			// NOTE: "depth" is a misnomer, it is actually the number of channels
			// NOTE: Need to change this to "channels" in the future

			float* input;
			float* output;
			float* kernel;

			// Convolution descriptor
			miopenConvolutionDescriptor_t convolutionDescriptor;

			// Convolution algorithm
			miopenConvFwdAlgorithm_t convolutionAlgorithm;
			size_t workspaceSize = 0;
			void* workspace = nullptr;

			void init_kernels();
			void fill_guassian_kernel(int sourceChannel, int targetChannel, float mu, float sigma, float* h_kernel);

			void set_descriptors();
			
			void find_algorithm();
	};
}