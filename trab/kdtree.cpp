#include "kdtree.h"

KDTreeGreenhouse::KDTreeGreenhouse(data_type *data, array_size n_datapoints,
                                   int n_components)
    : n_datapoints{n_datapoints}, n_components{n_components} {
  bool should_delete_data = false;

  if (data == nullptr) {
    throw std::invalid_argument("Received a null dataset.");
  }

  std::vector<DataPoint> data_points =
      as_data_points(data, this->n_datapoints, this->n_components);
  // 1D representation of our KDTree, or nullptr (if not main process)
  data_type *tree = grow_kd_tree(data_points);

  // tree is nullptr if the branch assigned to this process was empty
  if (tree != nullptr)
    grown_kd_tree =
        convert_to_knodes(tree, grown_kdtree_size, this->n_components, 0, 1, 0);
  else {
    grown_kd_tree = new KNode<data_type>();
  }

  if (should_delete_data)
    delete[] data;
}

data_type *KDTreeGreenhouse::grow_kd_tree(std::vector<DataPoint> &data_points) {
#pragma omp parallel default(none) shared(n_parallel_workers, max_parallel_depth, surplus_workers, std::cout, growing_tree, tree_size, data_points)
  {
#pragma omp single
    {
      n_parallel_workers = get_n_parallel_workers();
      max_parallel_depth = compute_max_depth(n_parallel_workers);
      surplus_workers =
          compute_n_surplus_processes(n_parallel_workers, max_parallel_depth);

#ifdef DEBUG
      std::cout << "Starting parallel region with " << n_parallel_workers
                << " parallel workers." << std::endl;
#endif

      // we want to store the tree (in a temporary way) in an array whose size
      // is a powersum of two
      tree_size = powersum_of_two(n_datapoints, true);
      growing_tree = new std::optional<DataPoint>[tree_size];

      int starting_region_width;
      starting_region_width = 1;

      build_tree_single_core(data_points.begin(), data_points.end(), 0,
                             starting_region_width, 0, 0);
    }
  }

  if (!data_points.empty())
    return finalize();
  else
    return nullptr;
}

void KDTreeGreenhouse::build_tree_single_core(
    std::vector<DataPoint>::iterator first_data_point,
    std::vector<DataPoint>::iterator end_data_point, int depth,
    array_size region_width, array_size region_start_index,
    array_size branch_starting_index) {
  // this is equivalent to say that there is at most one data point in the
  // sequence
  if (first_data_point + 1 != end_data_point) {
    int dimension = select_splitting_dimension(depth, n_components);
    array_size split_point_idx =
        sort_and_split(first_data_point, end_data_point, dimension);

    growing_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*(first_data_point + split_point_idx))));

    array_size region_start_index_left, region_start_index_right;
    array_size branch_start_index_left, branch_start_index_right;

    region_start_index_left = region_start_index_right =
        region_start_index + region_width;

    // the width of the next level will be twice the width of the current
    // level
    region_width *= 2;

    branch_starting_index *= 2;
    branch_start_index_left = branch_starting_index;
    branch_start_index_right = branch_starting_index + 1;
    depth += 1;

#pragma omp task final(depth > max_parallel_depth + 1 || (depth == max_parallel_depth + 1 && get_rank() >= surplus_workers))
    {
#ifdef DEBUG
#ifdef USE_OMP
      std::cout << "Task assigned to thread " << omp_get_thread_num()
                << std::endl;
#endif
#endif
      // right
      build_tree_single_core(first_data_point + split_point_idx + 1, end_data_point, depth,
                             region_width, region_start_index_right,
                             branch_start_index_right);
    }
    // left
    if (split_point_idx > 0)
      build_tree_single_core(first_data_point, first_data_point + split_point_idx, depth, region_width,
                             region_start_index_left, branch_start_index_left);

#pragma omp taskwait
  } else {
    growing_tree[region_start_index + branch_starting_index].emplace(
        DataPoint(std::move(*first_data_point)));
  }
}
