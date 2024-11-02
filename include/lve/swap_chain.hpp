#pragma once

#include "lve/device.hpp"

#include <vulkan/vulkan.h>

#include <string>
#include <vector>


namespace lve {

    // This class is used to create and manage the swap chain
    // which is used to present images to the screen
    class LveSwapChain {

        public:
            static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

            LveSwapChain(LveDevice &deviceRef, VkExtent2D windowExtent);
            ~LveSwapChain();

            // Not copyable or movable
            LveSwapChain(const LveSwapChain &) = delete;
            void operator=(const LveSwapChain &) = delete;

            // Getters
            VkFramebuffer getFrameBuffer(int index) { return swapChainFramebuffers[index]; }
            VkRenderPass getRenderPass() { return renderPass; }
            VkImageView getImageView(int index) { return swapChainImageViews[index]; }
            size_t imageCount() { return swapChainImages.size(); }
            VkFormat getSwapChainImageFormat() { return swapChainImageFormat; }
            VkExtent2D getSwapChainExtent() { return swapChainExtent; }
            uint32_t width() { return swapChainExtent.width; }
            uint32_t height() { return swapChainExtent.height; }

            float extentAspectRatio() {
            return static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
            }
            VkFormat findDepthFormat();

            VkResult acquireNextImage(uint32_t *imageIndex);
            VkResult submitCommandBuffers(const VkCommandBuffer *buffers, uint32_t *imageIndex);

        private:
            void createSwapChain();
            void createImageViews();
            void createDepthResources();
            void createRenderPass();
            void createFramebuffers();
            void createSyncObjects();

            // Helper functions
            VkSurfaceFormatKHR chooseSwapSurfaceFormat(
                const std::vector<VkSurfaceFormatKHR> &availableFormats);
            VkPresentModeKHR chooseSwapPresentMode(
                const std::vector<VkPresentModeKHR> &availablePresentModes);
            VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);

            VkFormat swapChainImageFormat;
            VkExtent2D swapChainExtent;

            std::vector<VkFramebuffer> swapChainFramebuffers;
            VkRenderPass renderPass;

            // Depth resources
            std::vector<VkImage> depthImages;
            std::vector<VkDeviceMemory> depthImageMemorys;
            std::vector<VkImageView> depthImageViews;
            std::vector<VkImage> swapChainImages;
            std::vector<VkImageView> swapChainImageViews;

            LveDevice &device;
            VkExtent2D windowExtent;

            VkSwapchainKHR swapChain;

            // Synchronization
            std::vector<VkSemaphore> imageAvailableSemaphores;
            std::vector<VkSemaphore> renderFinishedSemaphores;
            std::vector<VkFence> inFlightFences;
            std::vector<VkFence> imagesInFlight;
            size_t currentFrame = 0;
    };
}
