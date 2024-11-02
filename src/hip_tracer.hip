#include "hip_tracer.hpp"
#include "htc/lenia_graph.hpp"
#include "htc/utils.hpp"

#include <vulkan/vulkan.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <stdexcept>


namespace htc {

    HipTracer::HipTracer(int width, int height, uint32_t outputBuffersCount, lve::LveDevice& lveDevice) :
        width(width), height(height), outputBuffersCount(outputBuffersCount), lveDevice(lveDevice) {

        // Create the output frame buffers
        createOutputFrameBuffers();

        // Create the HIP context
        CHECK_HIP_ERROR(hipInit(0));
        
        // Create the Compute Graph
        leniaGraph.emplace(width, height, outputFrameBuffers[0]);
    }

    HipTracer::~HipTracer() {
        for (uint32_t i = 0; i < outputBuffersCount; i++) {
            CHECK_HIP_ERROR(hipDestroyExternalMemory(hipExternalMemoryHandles[i]));

            vkDestroyBuffer(lveDevice.device(), interoperabilityBuffers[i], nullptr);
            vkFreeMemory(lveDevice.device(), interoperabilityMemories[i], nullptr);
        }
    }

    void HipTracer::createOutputFrameBuffers() {
        // Define the buffer size and extend the vectors
        VkDeviceSize bufferSize = sizeof(lve::Vertex) * width * height;

        outputFrameBuffers.resize(outputBuffersCount);
        hipExternalMemoryHandles.resize(outputBuffersCount);
        interoperabilityBuffers.resize(outputBuffersCount);
        interoperabilityMemories.resize(outputBuffersCount);

        // Load vkGetMemoryFdKHR manually (since it's not in the Vulkan headers)
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(lveDevice.device(), "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) {
            throw std::runtime_error("Failed to load vkGetMemoryFdKHR!");
        }

        for (uint32_t i = 0; i < outputBuffersCount; i++) {
            // Use createBuffer with external parameters
            lveDevice.createBuffer(
                bufferSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                interoperabilityBuffers[i],
                interoperabilityMemories[i],
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
            );

            // Get the file descriptor for the memory
            int fd;
            VkMemoryGetFdInfoKHR getFdInfo = {};
            getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            getFdInfo.memory = interoperabilityMemories[i];
            getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

            if (vkGetMemoryFdKHR(lveDevice.device(), &getFdInfo, &fd) != VK_SUCCESS) {
                throw std::runtime_error("Failed to get memory file descriptor!");
            }

            // Import the memory into HIP
            hipExternalMemoryHandleDesc extMemHandleDesc = {};
            extMemHandleDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
            extMemHandleDesc.handle.fd = fd;
            extMemHandleDesc.size = bufferSize;

            CHECK_HIP_ERROR(hipImportExternalMemory(&hipExternalMemoryHandles[i], &extMemHandleDesc));

            // Map the HIP memory to a usable buffer
            hipExternalMemoryBufferDesc bufferDesc = {};
            bufferDesc.offset = 0;
            bufferDesc.size = bufferSize;
            bufferDesc.flags = 0;

            CHECK_HIP_ERROR(hipExternalMemoryGetMappedBuffer((void**)&outputFrameBuffers[i], hipExternalMemoryHandles[i], &bufferDesc));
        }
    }

    std::vector<VkBuffer> HipTracer::bind() {
        return interoperabilityBuffers;
    }

    void HipTracer::getNextFrame(uint32_t outputBufferIndex) {
        // Step the Compute Graph and write the output to the output buffer
        leniaGraph->step(outputFrameBuffers[outputBufferIndex]);
    }
}