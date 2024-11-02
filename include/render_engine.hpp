#pragma once

#include "lve/window.hpp"
#include "lve/pipeline.hpp"
#include "lve/device.hpp"
#include "lve/swap_chain.hpp"
#include "lve/multiple_vertex_buffer.hpp"
#include "lve/utils.hpp"

#include "hip_tracer.hpp"

#include <memory>
#include <vector>
#include <thread>
#include <atomic>

#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512

#define WINDOW_NAME "Lenia with Vulkan and HIP"

#define FPS_COUNTER_DISPLAY_INTERVAL 5


namespace lve {

	// This class is responsible for managing all ressources and operations related to Vulkan
	// it gets its data from a HipTracer object and renders it to the screen
	class RenderEngine {

		public:
			// Window dimensions
			static constexpr int WIDTH = WINDOW_WIDTH;
			static constexpr int HEIGHT = WINDOW_HEIGHT;

			RenderEngine();
			~RenderEngine();

			// Not copyable or movable
			RenderEngine(const RenderEngine&) = delete;
			RenderEngine& operator=(const RenderEngine&) = delete;

			void run();

		private:
			void createVertexSupplier();
			void createPipelineLayout();
			void createPipeline();
			void createCommandBuffers();
			void updateVertexData();
			void drawFrame();

			void startUpdateThread();
			void stopUpdateThread();

			// Window and Vulkan objects
			LveWindow lveWindow{WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME};
			LveDevice lveDevice{lveWindow};
			LveSwapChain lveSwapChain{lveDevice, lveWindow.getExtent()};
			std::unique_ptr<LvePipeline> lvePipeline;
			VkPipelineLayout pipelineLayout;
			std::vector<VkCommandBuffer> commandBuffers;
			std::unique_ptr<LveMultipleVertexBuffer> lveMultipleVertexBuffer;

			// Main loop
			std::thread updateThread;
			std::atomic<bool> running{true};

			// FPS counter
			FPSCounter fpsCounter{FPS_COUNTER_DISPLAY_INTERVAL};

			// HipTracer object
			std::unique_ptr<htc::HipTracer> vertexSupplier;
	};
}
