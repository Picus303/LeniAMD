#pragma once

#include "htc/lenia_graph.hpp"

#include "lve/device.hpp"
#include "lve/utils.hpp"

#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <optional>


namespace htc {

	// This class is responsible for managing all operations related to image processing
	// using a HIP-accelerated graph of operations and outputting the results to a Vulkan buffer
	class HipTracer {

		public:

			HipTracer(int width, int height, uint32_t outputBuffersCount, lve::LveDevice& lveDevice);
			~HipTracer();

			// Not copyable or movable
			HipTracer(const HipTracer&) = delete;
			HipTracer& operator=(const HipTracer&) = delete;

			std::vector<VkBuffer> bind();
			void getNextFrame(uint32_t outputBufferIndex);

		private:

			void createOutputFrameBuffers();

			int width;
			int height;

			uint32_t outputBuffersCount;

			lve::LveDevice& lveDevice;

			std::vector<lve::Vertex*> outputFrameBuffers;

			std::vector<void*> hipExternalMemoryHandles;

			std::vector<VkBuffer> interoperabilityBuffers;
			std::vector<VkDeviceMemory> interoperabilityMemories;

			std::optional<LeniaGraph> leniaGraph;
	};
}