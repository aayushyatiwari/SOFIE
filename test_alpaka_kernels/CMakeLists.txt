cmake_minimum_required(VERSION 3.22)

list(APPEND CMAKE_PREFIX_PATH "/home/imaayush/code/alpaka/install/lib/cmake/alpaka")

project(MyProject CUDA CXX)

find_package(alpaka REQUIRED)
alpaka_add_executable(elu_kernel elu.cu)
target_link_libraries(elu_kernel PRIVATE alpaka::alpaka)
target_compile_definitions(elu_kernel PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)

alpaka_add_executable(relu_kernel relu.cu)
target_link_libraries(relu_kernel PRIVATE alpaka::alpaka)
target_compile_definitions(relu_kernel PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
file(GLOB SOURCES "*.cu")
