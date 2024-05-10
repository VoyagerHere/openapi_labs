#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <string_view>

#include <CL/sycl.hpp>

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const& e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
  };

static std::mt19937& mersenneInstance() {
  static std::random_device rd;
  static std::mt19937 mersenne(rd());
  return mersenne;
}

static void fillRandomly(std::vector<float>& arr) {
  std::uniform_real_distribution<> urd(1.0, 3.0);
  for (float& el : arr)
    el = urd(mersenneInstance());
}

static float vectorLength(const float* x, size_t n) {
  float s = 0;
  for (size_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return std::sqrt(s);
}

static float normAbs(const float* x0, const float* x1, size_t n) {
  float s = 0;
  for (size_t i = 0; i < n; i++) {
    s += (x0[i] - x1[i]) * (x0[i] - x1[i]);
  }
  return std::sqrt(s);
}

static float normRel(const float* x0, const float* x1, size_t n) {
  return normAbs(x0, x1, n) / vectorLength(x0, n);
}

static float deviationAbs(const float* a, const float* b, const float* x, int n) {
  float norm = 0;
  for (int i = 0; i < n; i++) {
    float s = 0;
    for (int j = 0; j < n; j++) {
      s += a[j * n + i] * x[j];
    }
    s -= b[i];
    norm += s * s;
  }
  return sqrt(norm);
}

static float deviationRel(const float* a, const float* b, const float* x, int n) {
  return deviationAbs(a, b, x, n) / vectorLength(b, n);
}

namespace utils {

  std::pair<std::vector<float>, std::vector<float>> generateEquationSystem(size_t rowsCount) {
    std::vector<float> matrix(rowsCount * rowsCount, 0.f);
    fillRandomly(matrix);
    std::uniform_real_distribution<> urd(rowsCount * 5.0, rowsCount * 5.0 + 2.0);
    for (size_t i = 0; i < rowsCount; i++)
      matrix[i * rowsCount + i] = urd(mersenneInstance());
    std::vector<float> col(rowsCount, 0.f);
    fillRandomly(col);
    return { matrix, col };
  }

  float norm(const float* x0, const float* x1, size_t n) {
    return normRel(x0, x1, n);
  }

  float norm(const std::vector<float>& x0, const std::vector<float>& x1) {
    return normRel(x0.data(), x1.data(), x0.size());
  }

  float deviation(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& x) {
    size_t n = x.size();
    return deviationRel(a.data(), b.data(), x.data(), n);
  }

} // namespace utils

namespace jacobi {

  struct CompResult {
    std::vector<float> x;
    double elapsed_all;
    double elapsed_kernel = 0;
    int iter;
    float accuracy;
  };

} // namespace jacobi

void calc_accessor(sycl::device d_selector, std::vector<float> A, std::vector<float> b, float accuracyTarget, int iterationsLimit, bool is_hot_run = true) {
  try {
    sycl::queue queue(d_selector, exception_handler,
      sycl::property_list{ sycl::property::queue::enable_profiling {} });

    jacobi::CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t global_size = b.size();
    size_t size_b = global_size * sizeof(float);
    size_t size_a = A.size() * sizeof(float);

    std::vector<float> x0 = b;
    std::vector<float> x1 = b;

    size_t globalSize = b.size();

    double begin = omp_get_wtime();

    sycl::buffer<float> buffer_a(A.data(), A.size());
    sycl::buffer<float> buffer_b(b.data(), b.size());

    do {
      {
        sycl::buffer<float> buffer_x0(x0.data(), b.size());
        sycl::buffer<float> buffer_x1(x1.data(), b.size());

        sycl::event event = queue.submit([&](sycl::handler& h) {
          auto accessor_a = buffer_a.get_access<sycl::access::mode::read>(h);
          auto accessor_b = buffer_b.get_access<sycl::access::mode::read>(h);
          sycl::accessor accessor_x0 = buffer_x0.get_access<sycl::access::mode::read_write>(h);
          sycl::accessor accessor_x1 = buffer_x1.get_access<sycl::access::mode::read_write>(h);

          if (result.iter % 2 == 1) {
            accessor_x0 = buffer_x1.get_access<sycl::access::mode::read_write>(h);
            accessor_x1 = buffer_x0.get_access<sycl::access::mode::read_write>(h);
          }

          h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
            int i = item.get_id(0);
            int n = item.get_range(0);
            float s = 0;
            for (int j = 0; j < n; j++)
              s += i != j ? accessor_a[j * n + i] * accessor_x0[j] : 0;
            accessor_x1[i] = (accessor_b[i] - s) / accessor_a[i * n + i];
            });
          });
        queue.wait();

        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        result.elapsed_kernel += (end - start) / 1e+6;
      }

      result.accuracy = utils::norm(x0, x1);
      result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);

    double end = omp_get_wtime();

    result.elapsed_all = (end - begin) * 1000.;
    result.x = x1;

    if (is_hot_run) {
      float deviation = utils::deviation(A, b, result.x);
      std::cout << " - Accessor\n" << "\tTime all : " << result.elapsed_all
        << " ms\n" << "\tTime kernel: " << result.elapsed_kernel << " ms\n" << "\tDeviation: " << deviation << "\n";
    }
  }
  catch (std::exception const& e) {

    std::cout << "An exception is caught.\n" << e.what() << "\n";
    std::terminate();
  }
}

void calc_shared(sycl::device d_selector, std::vector<float> A, std::vector<float> b, float accuracyTarget, int iterationsLimit, bool is_hot_run = true) {
  try {
    sycl::queue queue(d_selector, exception_handler,
      sycl::property_list{ sycl::property::queue::enable_profiling {} });

    jacobi::CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t global_size = b.size();
    size_t size_b = global_size * sizeof(float);
    size_t size_a = A.size() * sizeof(float);

    float* shared_a = sycl::malloc_shared<float>(A.size(), queue);
    float* shared_b = sycl::malloc_shared<float>(b.size(), queue);
    float* shared_x0 = sycl::malloc_shared<float>(b.size(), queue);
    float* shared_x1 = sycl::malloc_shared<float>(b.size(), queue);

    queue.memcpy(shared_a, A.data(), size_a).wait();
    queue.memcpy(shared_b, b.data(), size_b).wait();
    queue.memcpy(shared_x1, b.data(), size_b).wait();

    double begin = omp_get_wtime();
    do {
      // queue.memcpy(shared_x0, shared_x1, bSize).wait();
      std::swap(shared_x0, shared_x1);
      sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(global_size), [=](sycl::item<1> item) {
          int i = item.get_id(0);
          int n = item.get_range(0);
          float s = 0;
          for (int j = 0; j < n; j++)
            s += i != j ? shared_a[j * n + i] * shared_x0[j] : 0;
          shared_x1[i] = (shared_b[i] - s) / shared_a[i * n + i];
          });
        });
      queue.wait();

      auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
      result.elapsed_kernel += (end - start) / 1e+6;

      result.accuracy = utils::norm(shared_x0, shared_x1, global_size);
      result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    result.elapsed_all = (end - begin) * 1000.;

    result.x = std::vector<float>(global_size);
    queue.memcpy(result.x.data(), shared_x1, size_b);

    sycl::free(shared_a, queue);
    sycl::free(shared_b, queue);
    sycl::free(shared_x0, queue);
    sycl::free(shared_x1, queue);

    if (is_hot_run) {
      float deviation = utils::deviation(A, b, result.x);
      std::cout << " - Shared\n" << "\tTime all : " << result.elapsed_all
        << " ms\n" << "\tTime kernel: " << result.elapsed_kernel << " ms\n" << "\tDeviation: " << deviation << "\n";
    }
  }
  catch (std::exception const& e) {

    std::cout << "An exception is caught.\n" << e.what() << "\n";
    std::terminate();
  }
}

void calc_device(sycl::device d_selector, std::vector<float> A, std::vector<float> b, float accuracyTarget, int iterationsLimit, bool is_hot_run = true) {
  try {
    sycl::queue queue(d_selector, exception_handler,
      sycl::property_list{ sycl::property::queue::enable_profiling {} });

    jacobi::CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t global_size = b.size();
    size_t size_b = global_size * sizeof(float);
    size_t size_a = A.size() * sizeof(float);

    float* device_a = sycl::malloc_device<float>(A.size(), queue);
    float* device_b = sycl::malloc_device<float>(global_size, queue);
    float* device_x0 = sycl::malloc_device<float>(global_size, queue);
    float* device_x1 = sycl::malloc_device<float>(global_size, queue);

    std::vector<float> x0(global_size);
    std::vector<float> x1(global_size);

    queue.memcpy(device_a, A.data(), size_a).wait();
    queue.memcpy(device_b, b.data(), size_b).wait();
    queue.memcpy(device_x1, b.data(), size_b).wait();

    double begin = omp_get_wtime();
    do {
      std::swap(device_x0, device_x1);
      sycl::event event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(global_size), [=](sycl::item<1> item) {
          int i = item.get_id(0);
          int n = item.get_range(0);
          float s = 0;
          for (int j = 0; j < n; j++)
            s += i != j ? device_a[j * n + i] * device_x0[j] : 0;
          device_x1[i] = (device_b[i] - s) / device_a[i * n + i];
          });
        });
      queue.wait();

      auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
      result.elapsed_kernel += (end - start) / 1e+6;

      queue.memcpy(x0.data(), device_x0, size_b).wait();
      queue.memcpy(x1.data(), device_x1, size_b).wait();
      result.accuracy = utils::norm(x0, x1);
      result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    result.elapsed_all = (end - begin) * 1000.;
    result.x = x1;

    sycl::free(device_a, queue);
    sycl::free(device_b, queue);
    sycl::free(device_x0, queue);
    sycl::free(device_x1, queue);

    if (is_hot_run) {
      float deviation = utils::deviation(A, b, result.x);
      std::cout << " - Device\n" << "\tTime all : " << result.elapsed_all
        << " ms\n" << "\tTime kernel: " << result.elapsed_kernel << " ms\n" << "\tDeviation: " << deviation << "\n";
    }
  }
  catch (std::exception const& e) {

    std::cout << "An exception is caught.\n" << e.what() << "\n";
    std::terminate();
  }
}

int main(int argc, char* argv[]) {

  size_t rowsCount = 500;
  float accuracyTarget = 0.0001;
  int iterationsLimit = 100;

  if (argc != 4) {
    std::cout << "No arguments passed. Defaulting."
      << "\n\n";
  }
  else {
    rowsCount = static_cast<size_t>(std::atoi(argv[1]));
    accuracyTarget = std::atof(argv[2]);
    iterationsLimit = std::atoi(argv[3]);
  }

  sycl::device d(sycl::cpu_selector_v);
  std::cout << "Running on device: " << d.get_info<sycl::info::device::name>() << "\n";

  auto [A, b] = utils::generateEquationSystem(rowsCount);

  calc_accessor(d, A, b, accuracyTarget, iterationsLimit, false); // cold
  calc_accessor(d, A, b, accuracyTarget, iterationsLimit); // hot

  calc_shared(d, A, b, accuracyTarget, iterationsLimit, false); // cold
  calc_shared(d, A, b, accuracyTarget, iterationsLimit); // hot

  calc_device(d, A, b, accuracyTarget, iterationsLimit, false); // cold
  calc_device(d, A, b, accuracyTarget, iterationsLimit); // hot
}
