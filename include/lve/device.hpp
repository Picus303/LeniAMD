#pragma once

#include "lve/window.hpp"

#include <string>
#include <vector>


namespace lve {

    // Forward declarations
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    struct QueueFamilyIndices {
        uint32_t graphicsFamily;
        uint32_t presentFamily;
        bool graphicsFamilyHasValue = false;
        bool presentFamilyHasValue = false;
        bool isComplete() { return graphicsFamilyHasValue && presentFamilyHasValue; }
    };

    // This class is responsible for creating and managing the physical and logical Vulkan device
    // It handles all the interactions between the application and the hardware
    // NOTE: Most of Vulkan's initialization is done in this class
    class LveDevice {

        public:
            // Check if we are in debug mode
            #ifdef NDEBUG
            const bool enableValidationLayers = false;
            #else
            const bool enableValidationLayers = true;
            #endif

            LveDevice(LveWindow &window);
            ~LveDevice();

            // Not copyable or movable
            LveDevice(const LveDevice &) = delete;
            void operator=(const LveDevice &) = delete;
            LveDevice(LveDevice &&) = delete;
            LveDevice &operator=(LveDevice &&) = delete;

            // Getters
            VkCommandPool getCommandPool() { return commandPool; }
            VkDevice device() { return device_; }
            VkSurfaceKHR surface() { return surface_; }
            VkQueue graphicsQueue() { return graphicsQueue_; }
            VkQueue presentQueue() { return presentQueue_; }

            // Swap chain support
            SwapChainSupportDetails getSwapChainSupport() { return querySwapChainSupport(physicalDevice); }
            uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
            QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(physicalDevice); }
            VkFormat findSupportedFormat(
                const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

            // Buffer Helper Functions
            void createBuffer(
                VkDeviceSize size,
                VkBufferUsageFlags usage,
                VkMemoryPropertyFlags properties,
                VkBuffer &buffer,
                VkDeviceMemory &bufferMemory,
                VkExternalMemoryHandleTypeFlags handleTypes = 0);

            VkCommandBuffer beginSingleTimeCommands();
            void endSingleTimeCommands(VkCommandBuffer commandBuffer);
            void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
            void copyBufferToImage(
                VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t layerCount);

            void createImageWithInfo(
                const VkImageCreateInfo &imageInfo,
                VkMemoryPropertyFlags properties,
                VkImage &image,
                VkDeviceMemory &imageMemory);

            VkPhysicalDeviceProperties properties;

        private:
            // Initialization functions
            void createInstance();
            void setupDebugMessenger();
            void createSurface();
            void pickPhysicalDevice();
            void createLogicalDevice();
            void createCommandPool();

            // helper functions
            bool isDeviceSuitable(VkPhysicalDevice device);
            std::vector<const char *> getRequiredExtensions();
            bool checkValidationLayerSupport();
            QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
            void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
            void hasGflwRequiredInstanceExtensions();
            bool checkDeviceExtensionSupport(VkPhysicalDevice device);
            SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

            // Vulkan objects
            VkInstance instance;
            VkDebugUtilsMessengerEXT debugMessenger;
            VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
            LveWindow &window;
            VkCommandPool commandPool;

            VkDevice device_;
            VkSurfaceKHR surface_;
            VkQueue graphicsQueue_;
            VkQueue presentQueue_;

            // Constants and Extensions
            const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
            // NOTE: These extensions are installation specific and may not be available by default
            const std::vector<const char *> instanceExtensions = {
                VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
                VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            };
            // NOTE: Theses extensions are hardware specific and may not be supported by all devices
            const std::vector<const char *> deviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,      // Required for HIP interoperability
                VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,   // Required for HIP interoperability
            };
    };
}