#pragma once

#include "file_utils.h"
#include "kdtree.h"
#include "tree_printer.h"
#include "utils.h"

#include <optional>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#else
#include <chrono>
#include <cstdint>
#endif

inline data_type *read_file_serial(const std::string filename,
                                   std::size_t *size, int *dims) {
  // if we're using OpenMP there's no need to check that this is the main process
  return read_file(filename, size, dims);
}

inline void write_file_serial(const std::string &filename,
                              KNode<data_type> *root, const int dims) {
  // if we're using OpenMP there's no need to check that this is the main process
  write_file(filename, root, dims);
}

inline double get_time() {
#ifdef USE_OMP
  return omp_get_wtime();
#else
  return (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::high_resolution_clock::now()
                          .time_since_epoch())
                      .count()) /
         1000000000;
#endif
}

template <typename T> inline void log_message(T obj) {
  std::cout << obj << std::endl;
}

inline void init_parallel_environment(int *argc, char ***argv) {
  // No initialization needed for OpenMP or serial execution
}

inline void finalize_parallel_environment() {
  // No finalization needed for OpenMP or serial execution
}

#ifdef TEST
inline bool initialize_test(KNode<data_type> *tree, int n_components) {
  std::vector<std::optional<data_type>> *constraints =
      new std::vector<std::optional<data_type>>(n_components * 2);
  for (int i = 0; i < n_components; ++i) {
    // an empty optional
    std::optional<data_type> low;
    (*constraints)[i * 2] = low;

    // an other empty optional
    std::optional<data_type> high;
    (*constraints)[i * 2 + 1] = high;
  }
  return test_kd_tree(tree, constraints, 0);
}

inline bool test(KNode<data_type> *tree, int n_components) {
  return initialize_test(tree, n_components);
}
#endif

