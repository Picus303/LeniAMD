#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <iostream>

// Macros for handling HIP and MIOpen errors

#define CHECK_HIP_ERROR(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) << " at line " << __LINE__ << " in " << __FILE__ << std::endl; \
        exit(1); \
    }

#define CHECK_MIOPEN_ERROR(status) \
    if (status != miopenStatusSuccess) { \
        std::cerr << "MIOpen Error: " << status << " at line " << __LINE__ << " in " << __FILE__ << std::endl; \
        exit(1); \
    }