#include <alpaka/alpaka.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace alpaka;

struct ReluKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            float* data,
            std::size_t numElements) const
    {
        printf("elements=%u first=%u stride=%u\n",
            (uint32_t)alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0],
            (uint32_t)alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0],
            (uint32_t)alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0]);
        printf("kernel running, numElements=%u\n", numElements);
        for (auto i : alpaka::uniformElements(acc, numElements))
        {
            printf("index %u\n", (unsigned)i);
            data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
        }
    }
};


/*
struct ReluKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
            TAcc const& acc,
            float* data,
            std::size_t numElements
            ) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if (idx < numElements)
        {
            printf("thread hitting index %u\n", (unsigned)idx);
            data[idx] = (data[idx] > 0.0f) ? data[idx] : 0.0f;
        }
    }
};

*/
auto main() -> int
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
    auto platform = alpaka::Platform<Acc>{};
    auto device   = alpaka::getDevByIdx(platform, 0);
    auto queue    = alpaka::Queue<Acc, alpaka::Blocking>{device};
    auto hostPlatform = alpaka::PlatformCpu{};
    auto hostDev      = alpaka::getDevByIdx(hostPlatform, 0);

    std::vector<float> hostData = {-2.f, -1.f, 0.f, 1.f, 2.f};
    Idx n = static_cast<Idx>(hostData.size());

    auto devBuf  = alpaka::allocBuf<float, Idx>(device, n);
    auto hostView = alpaka::createView(hostDev, hostData.data(), n);
    alpaka::memcpy(queue, devBuf, hostView);

    ReluKernel kernel;
    auto workDiv = alpaka::getValidWorkDiv(
        alpaka::KernelCfg<Acc>{
            alpaka::Vec<Dim, Idx>{n},
            alpaka::Vec<Dim, Idx>{static_cast<Idx>(1)}
        },
        device, kernel,
        alpaka::getPtrNative(devBuf), n);

    alpaka::exec<Acc>(queue, workDiv, kernel,
        alpaka::getPtrNative(devBuf), n);

    alpaka::memcpy(queue, hostView, devBuf);
    alpaka::wait(queue);

    std::cout << "ReLU output:\n";
    for (Idx i = 0; i < n; i++)
        std::cout << "  " << hostData[i] << "\n";

    return 0;
}
