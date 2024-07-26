#pragma once

#include "data_point.h"
#include "knode.h"
#include "process_utils.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <optional>
#include <unistd.h>
#include <vector>

#include <omp.h>

class KDTreeGreenhouse {
private:
  // number of datapoints managed by this class (might vary if some portions are
  // delegated to other processes).
  array_size n_datapoints;
  // number of components for each datapoint.
  int n_components;

  // the depth in the whole tree that this kd-tree is located at. used to
  // determine which child process are to be used in further parallel
  // splittings.
  int starting_depth = 0;

  // number of MPI processes/OMP threads available
  int n_parallel_workers = -1;

  // maximum depth of the tree at which we can parallelize using OMP. after
  // this depth no more right-branches can be assigned to non-surplus
  // workers.
  int max_parallel_depth = 0;

  // number of additional processes/threads that are not enough to parallelize
  // an entire level of the tree, and are therefore assigned
  // left-to-right until there are no more left.
  int surplus_workers = 0;

  // number of items assigned to a particular core (i.e. non-parallelizable with
  // MPI).
  array_size tree_size = 0;

  // DataPoints assigned to this process/core. see also
  // build_tree_single_core().
  std::optional<DataPoint> *growing_tree = nullptr;

  // full size (different than the number of datapoints) of the kd-tree grown by
  // this greenhouse.
  array_size grown_kdtree_size = 0;
  // root node of the kd-tree grown by this greenhouse.
  KNode<data_type> *grown_kd_tree = nullptr;

  data_type *grow_kd_tree(std::vector<DataPoint> &data_points);

  void build_tree_single_core(std::vector<DataPoint>::iterator first_data_point,
                              std::vector<DataPoint>::iterator end_data_point,
                              int depth, array_size region_width,
                              array_size region_start_index,
                              array_size branch_starting_index);
  data_type *finalize();

public:
  KDTreeGreenhouse(data_type *data, array_size n_datapoints, int n_components);
  ~KDTreeGreenhouse() { delete grown_kd_tree; }

  KNode<data_type> &&extract_grown_kdtree() {
    return std::move(*grown_kd_tree);
  }
  array_size get_grown_kdtree_size() { return grown_kdtree_size; }
};

