#pragma once

#include "htc/convolution_manager.hpp"

#include "lve/utils.hpp"

#include <hip/hip_runtime.h>
#include <optional>

#define CHANNELS 3


namespace htc {

	// This class is responsible for managing the ressources and the execution of the graph
	// that represents the Lenia simulation in the GPU
	class LeniaGraph {

		public:

			LeniaGraph(int width, int height, lve::Vertex* templateVertexArray);
			~LeniaGraph();

			// Not copyable or movable
			LeniaGraph(const LeniaGraph&) = delete;
			LeniaGraph& operator=(const LeniaGraph&) = delete;

			void step(lve::Vertex* outputVertexArray);

		private:

			// MIOPEN convolution manager
			std::optional<ConvolutionManager> convolutionManager;

			hipStream_t stream;

			// Graph nodes parameters
			hipHostNodeParams convolutionNodeParams;
			hipKernelNodeParams updateNodeParams;
			hipKernelNodeParams colorNodeParams;

			// Graph nodes
			hipGraphNode_t convolutionNode;
			hipGraphNode_t updateNode;
			hipGraphNode_t colorNode;

			hipGraph_t graph;
			hipGraphExec_t graphExec;

			int width;
			int height;

			int depth = CHANNELS;

			// State of the simulation
			float* d_state;
			float* d_intermediate;

			void init_state();

			void createConvolutionNode();
			void createUpdateNode();
			void createColorNode(lve::Vertex* templateVertexArray);
	};
}