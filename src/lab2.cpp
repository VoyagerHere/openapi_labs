#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string_view>

#include <CL/sycl.hpp>

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const &e : e_list) {
        try {
            std::rethrow_exception(e);
        } catch (std::exception const &e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

int main(int argc, char *argv[]) {

    int stepsCount = 1000;

    if (argc != 2) {
        std::cout << "No arguments passed. Default = 1000"
                  << "\n\n";
    } else {
        stepsCount = std::atoi(argv[1]);
    }

    constexpr size_t groupSize = 16;
    double expected = 2.f * std::pow(std::sin(0.5f), 2.f) * std::sin(1.f);
    double dx = 1.f / stepsCount;
    double dy = 1.f / stepsCount;

    size_t groupsCount = stepsCount / groupSize + 1;
    std::vector<double> result(groupsCount * groupsCount, 0);

    auto d_selector{sycl::default_selector_v};

    try {
        sycl::queue queue(d_selector, exception_handler,
                          sycl::property_list{sycl::property::queue::enable_profiling{}});

        std::cout << "Running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Size: " << stepsCount << " x " << stepsCount << "\n\n";

        {
            uint64_t start = 0;
            uint64_t end = 0;

            sycl::range<1> num_items{result.size()};
            sycl::buffer<double> buffer(result.data(), num_items);
            buffer.set_write_back(true);

            sycl::event event = queue.submit([&](sycl::handler &h) {
                sycl::accessor accessor(buffer, h, sycl::write_only, sycl::no_init);

                h.parallel_for(
                    sycl::nd_range<2>(sycl::range<2>(stepsCount, stepsCount), sycl::range<2>(groupSize, groupSize)),
                    [=](sycl::nd_item<2> item) {
                        double x = dx * (item.get_global_id(0) + 0.5);
                        double y = dy * (item.get_global_id(1) + 0.5);
                        double value = sycl::sin(x) * sycl::cos(y);
                        double sum = sycl::reduce_over_group(item.get_group(), value, std::plus<double>());
                        if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
                            accessor[item.get_group(0) * item.get_group_range(0) + item.get_group(1)] = sum;
                        }
                    });
            });
            queue.wait();

            start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

            std::cout << "Kernel time: " << (end - start) / 1e+6 << " ms"
                      << "\n\n";
        }
    } catch (std::exception const &e) {
        std::cout << "An exception is caught for integration.\n";
        std::terminate();
    }

    double computed = std::accumulate(result.begin(), result.end(), 0.f) * dx * dy;
    result.clear();

    std::cout << "Integration successfully completed on device.\n";

    std::cout << "Expected: " << expected << "\n";
    std::cout << "Computed: " << computed << "\n";
    std::cout << "Error: " << std::abs(computed - expected) << "\n";
}