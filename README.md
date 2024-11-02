# LENIAMD: An Highly-Optimized Cellular Automaton LENIA for GPU

**Quick Summary:** This project implements LENIA, a complex cellular automaton, leveraging the interoperability between HIP and Vulkan to achieve exceptional GPU computation efficiency.

## 🎯 Context and Motivation

Simulating cellular automata, particularly LENIA, poses a challenge in terms of massive computations and fine-grained GPU resource management. This project marks my first major achievement in low-level GPU computation and has allowed me to explore and master advanced technologies (HIP, Vulkan) to write optimised, hardware specific code, for custom use-cases.

**Note :** This project was optimised specifically for my GPU, a Radeon 6800XT, and might not be perfectly suited for other GPUs.

## ⚙️ Key Features

- **Low-Level GPU Computation with HIP and Vulkan**: The implementation relies on HIP for low-level computation optimization, complemented by Vulkan, ensuring a high-performance GPU computation pipeline.
- **Memory Interoperability**: Interoperability between HIP and Vulkan to make optimal use of GPU resources, enabling unprecedented smoothness and execution speed for LENIA.
- **Maximum Performance**: Manual optimization and management of GPU operations that allow to make the most of the silicon.

## 🛠️ Technologies and Best Practices Used

- **Languages and Frameworks**: C++, HIP, Vulkan
- **Libraries and Tools**: GLFW 3, GLM
- **Advanced Development Practices**:
  - Fine-tuned GPU memory and threads management
  - Specific optimizations for parallel processing
  - Performance testing to measure and adjust efficiency gains

## 🚀 Build and Run

For those who wish to explore the code, here are the steps to install and run the project.

**Note :** This project requires you to have HIP/ROCm and the Vulkan SDK installed.

```bash
# Clone the repo
git clone https://github.com/Picus303/LeniAMD.git

# Enter the folder
mkdir LeniAMD/build && cd LeniAMD/build

# Compile
cmake ..
make

# Run the simulation
./lenia
```