#include "lve/utils.hpp"

#include <stdio.h>


namespace lve {

	FPSCounter::FPSCounter(int printInterval) : fpsInterval{printInterval} {
		// Set the lastFPSUpdate to the current time
		lastFPSUpdate = std::chrono::high_resolution_clock::now();
	}

	void FPSCounter::update() {
		// Increment the frame count
		auto currentTime = std::chrono::high_resolution_clock::now();
		frameCount++;

		// Calculate the elapsed time since the last update
		std::chrono::duration<float> elapsedTime = currentTime - lastFPSUpdate;
		if (elapsedTime.count() >= fpsInterval) {
			float fps = frameCount / elapsedTime.count();
			printf("FPS: %.2f\n", fps);

			// Reset the counters
			frameCount = 0;
			lastFPSUpdate = currentTime;
		}
	}

	// Define the vertex format: Position [0] and Color [1]
	std::vector<VkVertexInputBindingDescription> Vertex::getBindingDescriptions() {
		std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
		bindingDescriptions[0].binding = 0;
		bindingDescriptions[0].stride = sizeof(Vertex);
		bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescriptions;
	}

	std::vector<VkVertexInputAttributeDescription> Vertex::getAttributeDescriptions() {
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
}