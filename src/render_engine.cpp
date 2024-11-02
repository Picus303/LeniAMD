#include "render_engine.hpp"
#include "lve/utils.hpp"

#include "hip_tracer.hpp"

#include <stdexcept>
#include <stdio.h>
#include <array>
#include <cmath>


namespace lve {

	RenderEngine::RenderEngine() {
		createVertexSupplier();
		createPipelineLayout();
		createPipeline();
		createCommandBuffers();
		startUpdateThread();
	}

	RenderEngine::~RenderEngine() {
		vkDestroyPipelineLayout(lveDevice.device(), pipelineLayout, nullptr);
	}

	void RenderEngine::run() {
		while (!lveWindow.shouldClose()) {
			glfwPollEvents();
			drawFrame();

			fpsCounter.update();
		}

		// Wait for all operations to finish before cleaning up
		// If not done, we might remove resources that are still in use
		stopUpdateThread();
		lveMultipleVertexBuffer->exit();
		vkDeviceWaitIdle(lveDevice.device());
	}

	void RenderEngine::createVertexSupplier() {
		// Create the vertex supplier and the multiple vertex buffer
		uint32_t vertexBuffersCount = lveSwapChain.imageCount();
		vertexSupplier = std::make_unique<htc::HipTracer>(WIDTH, HEIGHT, vertexBuffersCount, lveDevice);
		lveMultipleVertexBuffer = std::make_unique<LveMultipleVertexBuffer>(lveDevice, vertexSupplier->bind(), vertexBuffersCount, WIDTH * HEIGHT);
	}

	void RenderEngine::createPipelineLayout() {
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(lveDevice.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}
	}

	void RenderEngine::createPipeline() {
		// Load the shaders and create the rendering pipeline
		auto pipelineConfig = LvePipeline::defaultPipelineConfigInfo(lveSwapChain.width(), lveSwapChain.height());
		pipelineConfig.renderPass = lveSwapChain.getRenderPass();
		pipelineConfig.pipelineLayout = pipelineLayout;
		lvePipeline = std::make_unique<LvePipeline>(
			lveDevice,
			"../shaders/simple_shader.vert.spv",
			"../shaders/simple_shader.frag.spv",
			pipelineConfig
		);
	}

	void RenderEngine::createCommandBuffers() {
		// Resize command buffers to hold one for each swap chain image
		// Then bind them to the rendering pipeline and the vertex buffer
		commandBuffers.resize(lveSwapChain.imageCount());
		VkCommandBufferAllocateInfo allocateInfo{};
		allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocateInfo.commandPool = lveDevice.getCommandPool();
		allocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

		if (vkAllocateCommandBuffers(lveDevice.device(), &allocateInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (int i = 0; i < static_cast<int>(commandBuffers.size()); i++) {
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			
			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = lveSwapChain.getRenderPass();
			renderPassInfo.framebuffer = lveSwapChain.getFrameBuffer(i);

			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = lveSwapChain.getSwapChainExtent();

			std::array<VkClearValue, 2> clearValues{};
			clearValues[0].color = {0.01f, 0.01f, 0.01f, 1.0f};
			clearValues[1].depthStencil = {1.0f, 0};

			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			lvePipeline->bind(commandBuffers[i]);
			lveMultipleVertexBuffer->bind(commandBuffers[i], i);
			lveMultipleVertexBuffer->draw(commandBuffers[i]);

			vkCmdEndRenderPass(commandBuffers[i]);
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	void RenderEngine::updateVertexData() {
		// Get the next available write buffer and update it
		uint32_t writeBufferIndex = lveMultipleVertexBuffer->getAvailableWriteBuffer();
		vertexSupplier->getNextFrame(writeBufferIndex);
		lveMultipleVertexBuffer->setReadBufferAvailable(writeBufferIndex);
	}

	void RenderEngine::drawFrame() {
		// Get the next available read buffer and submit it to the rendering pipeline
		uint32_t readBufferIndex = lveMultipleVertexBuffer->getAvailableReadBuffer();
		uint32_t imageIndex;

		auto result = lveSwapChain.acquireNextImage(&imageIndex);
		if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire next image!");
		}

		result = lveSwapChain.submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to submit command buffer!");
		}

		lveMultipleVertexBuffer->setWriteBufferAvailable(readBufferIndex);
	}

	void RenderEngine::startUpdateThread() {
		// This thread will update the vertex data in the GPU
		updateThread = std::thread([this]() {
			while (running.load()) {
				updateVertexData();
			}
		});
	}

	void RenderEngine::stopUpdateThread() {
		// Stop the thread and wait for it to finish
		running.store(false);
		if (updateThread.joinable()) {
			updateThread.join();
		}
	}
}