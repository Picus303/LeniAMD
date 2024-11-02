# pragma once

#include "lve/device.hpp"
#include "lve/utils.hpp"

#include <condition_variable>
#include <vector>
#include <mutex>
#include <deque>


namespace lve {

	// This class is used to contain the vertices of subsequent frames
	// and to manage the synchronization between the rendering and the updating of the vertices
	// it allows the redering pipeline and the frame generation pipeline to work in parallel
	// NOTE: There are as many vertex buffers as there are frames in the swap chain
	// NOTE: This way we don't need to rebind the vertex buffers every frame
	// NOTE: This may however lead to a slight increase in memory usage
	class LveMultipleVertexBuffer {
		public:

			LveMultipleVertexBuffer(LveDevice& device, std::vector<VkBuffer> vertexBuffers, uint32_t vertexBufferCount, uint32_t vertexCount);
			~LveMultipleVertexBuffer();

			// Not copyable or movable
			LveMultipleVertexBuffer(const LveMultipleVertexBuffer&) = delete;
			LveMultipleVertexBuffer& operator=(const LveMultipleVertexBuffer&) = delete;

			void bind(VkCommandBuffer commandBuffer, int vertexBufferIndex);
			void draw(VkCommandBuffer commandBuffer);

			// Called at the beginning of a render or generation pass
			uint32_t getAvailableReadBuffer();
			uint32_t getAvailableWriteBuffer();

			// Called at the end of a render or generation pass
			void setReadBufferAvailable(uint32_t bufferIndex);
			void setWriteBufferAvailable(uint32_t bufferIndex);

			void exit();

		private:

			bool running = true;

			LveDevice& lveDevice;

			// Vertex buffers
			std::vector<VkBuffer> vertexBuffers;
			uint32_t vertexBufferCount;
			uint32_t vertexCount;

			// Synchonization
			std::mutex readMutex;
			std::condition_variable readCondition;
			std::deque<uint32_t> availableReadBuffers;

			std::mutex writeMutex;
			std::condition_variable writeCondition;
			std::deque<uint32_t> availableWriteBuffers;
	};
}