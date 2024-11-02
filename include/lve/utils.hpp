#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vulkan/vulkan.h>
#include <chrono>
#include <vector>


namespace lve {

	// Define the vertex format
	struct Position {
		float x;
		float y;
	};

	struct Color {
		float r;
		float g;
		float b;
	};

	struct Vertex {
		Position position;
		Color color;

		static std::vector<VkVertexInputBindingDescription> getBindingDescriptions();
		static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
	};

	// This class is used to keep track of the frames per second
	// and print the FPS to the console at a given interval
	class FPSCounter {
		public:

			FPSCounter(int printInterval);
			void update();

		private:

			int frameCount = 0;
			const int fpsInterval;
			std::chrono::time_point<std::chrono::high_resolution_clock> lastFPSUpdate;
	};
}