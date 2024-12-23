#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include "NvOnnxParser.h"
#include<iostream>

// Logger class required for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

int main() {
    // Print TensorRT version
    std::cout << "TensorRT version: "
              << NV_TENSORRT_MAJOR << "."
              << NV_TENSORRT_MINOR << "."
              << NV_TENSORRT_PATCH << std::endl;

    // Try to create a TensorRT builder to verify the installation
    auto builder = nvinfer1::createInferBuilder(gLogger);
    if (builder == nullptr) {
        std::cerr << "Failed to create TensorRT builder!" << std::endl;
        return 1;
    }

    std::cout << "Successfully created TensorRT builder" << std::endl;

    // Clean up
    builder->reset();

    return 0;
}
