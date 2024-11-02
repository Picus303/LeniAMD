#include "lve/multiple_vertex_buffer.hpp"

#include <cstring>
#include <cassert>


namespace lve {

	LveMultipleVertexBuffer::LveMultipleVertexBuffer(LveDevice& device, std::vector<VkBuffer> vertexBuffers, uint32_t vertexBufferCount, uint32_t vertexCount) :
	lveDevice{device}, vertexBuffers{vertexBuffers}, vertexBufferCount{vertexBufferCount}, vertexCount{vertexCount} {
		for (uint32_t i = 0; i < vertexBufferCount; i++) {
			availableWriteBuffers.push_back(i);
		}
	}

	LveMultipleVertexBuffer::~LveMultipleVertexBuffer() {}

	void LveMultipleVertexBuffer::bind(VkCommandBuffer commandBuffer, int vertexBufferIndex) {
		// Bind the vertex buffer to the command buffer
		VkBuffer buffers[] = {vertexBuffers[vertexBufferIndex]};
		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);
	}

	void LveMultipleVertexBuffer::draw(VkCommandBuffer commandBuffer) {
		// Send the draw command to the command buffer
		vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
	}

	uint32_t LveMultipleVertexBuffer::getAvailableReadBuffer() {
		uint32_t readBufferIndex;

		{
			std::unique_lock<std::mutex> lock(readMutex);

			// Wait until there is a read buffer available
			if (availableReadBuffers.empty()) {
				readCondition.wait(lock, [this]() { return !availableReadBuffers.empty() || !running; });
				if (!running) {
					return 0;
				}
			}

			// Get the read buffer index
			readBufferIndex = availableReadBuffers.front();
			availableReadBuffers.pop_front();
		}

		return readBufferIndex;
	}

	uint32_t LveMultipleVertexBuffer::getAvailableWriteBuffer() {
		uint32_t writeBufferIndex;
		
		{
			std::unique_lock<std::mutex> lock(writeMutex);

			// Wait until there is a write buffer available
			if (availableWriteBuffers.empty()) {
				writeCondition.wait(lock, [this]() { return !availableWriteBuffers.empty() || !running; });
				if (!running) {
					return 0;
				}
			}

			// Get the write buffer index
			writeBufferIndex = availableWriteBuffers.front();
			availableWriteBuffers.pop_front();
		}

		return writeBufferIndex;
	}

	void LveMultipleVertexBuffer::setReadBufferAvailable(uint32_t bufferIndex) {
		{
			// Add the read buffer index to the available read buffers
			std::unique_lock<std::mutex> lock(readMutex);
			availableReadBuffers.push_back(bufferIndex);
		}
		// Notify the waiting threads that a read buffer is available
		readCondition.notify_one();
	}

	void LveMultipleVertexBuffer::setWriteBufferAvailable(uint32_t bufferIndex) {
		{
			// Add the write buffer index to the available write buffers
			std::unique_lock<std::mutex> lock(writeMutex);
			availableWriteBuffers.push_back(bufferIndex);
		}
		// Notify the waiting threads that a write buffer is available
		writeCondition.notify_one();
	}

	void LveMultipleVertexBuffer::exit() {
		// Notify all the waiting threads that the program is exiting
		// Use case: the rendering thread is waiting for a read buffer to be available
		// but the frame generation thread has already exited => infinite wait
		running = false;
		readCondition.notify_all();
		writeCondition.notify_all();
	}
}