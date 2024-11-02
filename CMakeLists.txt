cmake_minimum_required(VERSION 3.10)

project(lenia LANGUAGES CXX HIP)

# Add required packages
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(HIP REQUIRED)
find_package(miopen REQUIRED)

# Add source files
file(GLOB_RECURSE CXX_SOURCES src/*.cpp)
file(GLOB_RECURSE HIP_SOURCES src/*.hip)

add_executable(lenia ${CXX_SOURCES} ${HIP_SOURCES})

# Add include directories
target_include_directories(lenia PRIVATE ${Vulkan_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS} ${HIP_INCLUDE_DIRS} include)
target_link_libraries(lenia PRIVATE Vulkan::Vulkan glfw MIOpen)

# Compile with all warnings and optimizations
target_compile_options(lenia PRIVATE -Wall -Wextra -pedantic -O3)

# Compile shaders
add_custom_target(compile_shaders ALL
	COMMAND sh ${CMAKE_SOURCE_DIR}/shaders/compile.sh
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/shaders/
	COMMENT "Compiling shaders"
)

add_dependencies(lenia compile_shaders)

# Set HIP platform
set(HIP_PLATFORM amd)
add_definitions(-D__HIP_PLATFORM_AMD__)