#include "htc/lenia_graph.hpp"
#include "htc/convolution_manager.hpp"
#include "htc/kernels.hpp"
#include "htc/utils.hpp"

#include "lve/utils.hpp"

#include <hip/hip_runtime.h>
#include <random>


namespace htc {

	LeniaGraph::LeniaGraph(int width, int height, lve::Vertex* templateVertexArray) :
		width(width), height(height) {

		// Allocate memory for the state and intermediate arrays
		CHECK_HIP_ERROR(hipMalloc(&d_state, width * height * depth * sizeof(float)));
		CHECK_HIP_ERROR(hipMalloc(&d_intermediate, width * height * depth * sizeof(float)));

		// Create a stream
		CHECK_HIP_ERROR(hipStreamCreate(&stream));

		// Create the computation graph
		CHECK_HIP_ERROR(hipGraphCreate(&graph, 0));

		// Initialize Lenia with random values
		init_state();

		// Create the graph nodes
		createConvolutionNode();
		createUpdateNode();
		createColorNode(templateVertexArray);

		// NOTE: The convolution node is a host node because it doesn't execute any kernel
		// NOTE: but instead calls a MIOPEN function to perform the convolution

		// Instanciate the graph (compile it)
		CHECK_HIP_ERROR(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
	}

	LeniaGraph::~LeniaGraph() {
		// Make sure the computations are done
		CHECK_HIP_ERROR(hipStreamSynchronize(stream));

		// Free the resources
		CHECK_HIP_ERROR(hipGraphExecDestroy(graphExec));
		CHECK_HIP_ERROR(hipGraphDestroy(graph));
		CHECK_HIP_ERROR(hipStreamDestroy(stream));

		CHECK_HIP_ERROR(hipFree(d_intermediate));
		CHECK_HIP_ERROR(hipFree(d_state));
	}

	void LeniaGraph::init_state() {
		// Initialize the state with random values
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(0.0, 1.0);

		float* h_state = new float[width * height * depth];

		for (int i = 0; i < width * height * depth; i++) {
			h_state[i] = dis(gen);
		}

		// Copy the state to the device
		CHECK_HIP_ERROR(hipMemcpy(d_state, h_state, width * height * depth * sizeof(float), hipMemcpyHostToDevice));

		delete[] h_state;
	}

	void LeniaGraph::createConvolutionNode() {
		// Create the convolution manager
		convolutionManager.emplace(width, height, depth, d_state, d_intermediate);

		// Create the convolution node
		convolutionNodeParams = {};
		convolutionNodeParams.fn = [](void* userData) {
			ConvolutionManager* convolutionManager = static_cast<ConvolutionManager*>(userData);
			convolutionManager->runConvolution();
		};
		convolutionNodeParams.userData = &convolutionManager;

		CHECK_HIP_ERROR(hipGraphAddHostNode(&convolutionNode, graph, nullptr, 0, &convolutionNodeParams));
	}

	void LeniaGraph::createUpdateNode() {
		// Define block and grid dimensions
		dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
						(height + blockDim.y - 1) / blockDim.y,
						(depth + blockDim.z - 1) / blockDim.z);

		// Define the node parameters
		void* kernelParams[] = { (void*)&width, (void*)&height, (void*)&depth, (void*)&d_state, (void*)&d_intermediate };

		updateNodeParams = {};
		updateNodeParams.func = (void*)updateKernel;
		updateNodeParams.blockDim = blockDim;
		updateNodeParams.gridDim = gridDim;
		updateNodeParams.sharedMemBytes = 0;
		updateNodeParams.kernelParams = kernelParams;
		updateNodeParams.extra = nullptr;

		CHECK_HIP_ERROR(hipGraphAddKernelNode(&updateNode, graph, &convolutionNode, 1, &updateNodeParams));
	}

	void LeniaGraph::createColorNode(lve::Vertex* templateVertexArray) {
		// Define block and grid dimensions
		dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
						(height + blockDim.y - 1) / blockDim.y);

		// Define the node parameters
		void* kernelParams[] = { (void*)&width, (void*)&height, (void*)&d_state, (void*)&templateVertexArray };

		colorNodeParams = {};
		colorNodeParams.func = (void*)colorKernel;
		colorNodeParams.blockDim = blockDim;
		colorNodeParams.gridDim = gridDim;
		colorNodeParams.sharedMemBytes = 0;
		colorNodeParams.kernelParams = kernelParams;
		colorNodeParams.extra = nullptr;

		CHECK_HIP_ERROR(hipGraphAddKernelNode(&colorNode, graph, &updateNode, 1, &colorNodeParams));
	}

	void LeniaGraph::step(lve::Vertex* outputVertexArray) {
		// Redefine colorNodeParams parameters to output the result to the outputVertexArray
		void* kernelParams[] = { (void*)&width, (void*)&height, (void*)&d_state, (void*)&outputVertexArray };
		colorNodeParams.kernelParams = kernelParams;
		CHECK_HIP_ERROR(hipGraphExecKernelNodeSetParams(graphExec, colorNode, &colorNodeParams));

		// Execute the graph and wait for it to finish
		CHECK_HIP_ERROR(hipGraphLaunch(graphExec, stream));
		CHECK_HIP_ERROR(hipStreamSynchronize(stream));
	}
}