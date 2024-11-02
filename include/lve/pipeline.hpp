#pragma once

#include "lve/device.hpp"

#include <string>
#include <vector>


namespace lve {

	// Pipeline configuration struct
	struct PipelineConfigInfo {
		VkViewport viewport;
		VkRect2D scissor;
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
		VkPipelineRasterizationStateCreateInfo rasterizationInfo;
		VkPipelineMultisampleStateCreateInfo multisampleInfo;
		VkPipelineColorBlendAttachmentState colorBlendAttachment;
		VkPipelineColorBlendStateCreateInfo colorBlendInfo;
		VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
		VkPipelineLayout pipelineLayout = nullptr;
		VkRenderPass renderPass = nullptr;
		uint32_t subpass = 0;
	};

	// This class manages the graphics pipeline
	// it's mainly used as a wrapper around the VkPipeline object
	// to hide all the configuration details
	class LvePipeline {

		public:
			LvePipeline(
				LveDevice& device,
				const std::string& vertFilepath,
				const std::string& fragFilepath,
				const PipelineConfigInfo& configInfo);
			
			~LvePipeline();

			// Not copyable or movable
			LvePipeline(const LvePipeline&) = delete;
			void operator=(const LvePipeline&) = delete;

			void bind(VkCommandBuffer commandBuffer);

			static PipelineConfigInfo defaultPipelineConfigInfo(uint32_t width, uint32_t height);

		private:
			static std::vector<char> readFile(const std::string& filepath);

			void createGraphicsPipeline(
				const std::string& vertFilepath,
				const std::string& fragFilepath,
				const PipelineConfigInfo& configInfo);

			void createShaderModule(const std::vector<char>& code, VkShaderModule* shaderModule);

			LveDevice& lveDevice;
			VkPipeline graphicsPipeline;
			VkShaderModule vertShaderModule;
			VkShaderModule fragShaderModule;
	};
}