#include "utils.h"

int powersum_of_two(array_size n, bool greater) {
  // value of the current power of 2
  int current_level_magnitude = 2;
  // current value of the sum, and previous value
  array_size sum = 1;
  array_size prev_sum = 0;

  while (sum < n) {
    prev_sum = sum;
    sum += current_level_magnitude;
    current_level_magnitude *= 2;
  }
  return greater ? sum : prev_sum;
}

data_type *unpack_array(std::vector<DataPoint>::iterator first_point,
                        std::vector<DataPoint>::iterator last_point,
                        int n_components) {
  data_type *unpacked =
      new data_type[(last_point - first_point) * n_components];
  unpack_array(unpacked, first_point, last_point, n_components);
  return unpacked;
}

void unpack_array(data_type *dest, std::vector<DataPoint>::iterator first_point,
                  std::vector<DataPoint>::iterator last_point,
                  int n_components) {
  int offset = 0;
  for (auto i = first_point; i != last_point; ++i) {
    (*i).copy_to_array(dest + offset, n_components);
    offset += n_components;
  }
}

data_type *unpack_optional_array(std::optional<DataPoint> *array,
                                 array_size n_datapoints, int n_components,
                                 data_type fallback_value) {
  data_type *unpacked = new data_type[n_datapoints * n_components];
  for (array_size i = 0; i < n_datapoints; ++i) {
    if (array[i].has_value()) {
      DataPoint dp = std::move(*array[i]);
      dp.copy_to_array(unpacked + i * n_components, n_components);
    } else {
      std::fill_n(unpacked + i * n_components, n_components, fallback_value);
    }
  }
  return unpacked;
}

/*
  This function rearranges branch1 and branch2 into dest such that we first
  take 1 node from branch1 and 1 node from branch2, then 2 nodes from branch1
  and 2 nodes from branch2, then 4 nodes from branch1 and 4 nodes from
  branch2..

  Note that this function is dimensions-safe (i.e. copies all the dimensions).

  Remember to add a split point before this function call (if you need to).
*/
void merge_kd_trees(data_type *dest, data_type *branch1, data_type *branch2,
                    array_size branches_size, int n_components) {
  array_size already_added = 0;
  // number of nodes in each branch (left and right) at the current level of
  // the tree
  array_size nodes = 1;
  while (already_added < 2 * branches_size) {
    // we put into the three what's inside the left subtree
    std::memcpy(dest, branch1, nodes * n_components * sizeof(data_type));
    branch1 += nodes * n_components;
    dest += nodes * n_components;

    // we put into the three what's inside the right subtree
    std::memcpy(dest, branch2, nodes * n_components * sizeof(data_type));
    branch2 += nodes * n_components;
    dest += nodes * n_components;

    // we just added left and right branch
    already_added += nodes * 2;
    // the next level will have twice the number of nodes of the current level
    nodes *= 2;
  }
}

/*
    Convert the given tree to a linked list structure. This assumes that
    the given size is a powersum of two.

    - tree contains the array representation of the tree
    - size is the number of elements in `tree`
    - n_components is the number of components for each data point
    - current_level_start contains the index of the first element of tree
   which contains an element of the current node
    - current_level_nodes contains the number of elements in this level of the
        tree (each recursive call multiplies it by two)
    - start_offset contains the offset (starting from current_level_start) for
        the root node of the subtree represented by this recursive call.
*/
KNode<data_type> *convert_to_knodes(data_type *tree, array_size n_datapoints,
                                    int n_components,
                                    array_size current_level_start,
                                    array_size current_level_nodes,
                                    array_size start_offset) {
  array_size next_level_start =
      current_level_start + current_level_nodes * n_components;
  array_size next_level_nodes = current_level_nodes * 2;
  array_size next_start_offset = start_offset * 2;

  if (next_level_start < n_datapoints * n_components) {
    auto left =
        convert_to_knodes(tree, n_datapoints, n_components, next_level_start,
                          next_level_nodes, next_start_offset);
    auto right =
        convert_to_knodes(tree, n_datapoints, n_components, next_level_start,
                          next_level_nodes, next_start_offset + 1);

    return new KNode<data_type>(
        tree + current_level_start + start_offset * n_components, n_components,
        left, right, current_level_start == 0);
  } else
    return new KNode<data_type>(tree + current_level_start +
                                    start_offset * n_components,
                                n_components, nullptr, nullptr, false);
}

array_size sort_and_split(DataPoint *array, array_size n_datapoints, int axis) {
  // the second part of median_idx is needed to unbalance the split towards
  // the left region (which is the one which may parallelize with the highest
  // probability).
  array_size median_idx = n_datapoints / 2 - 1 * ((n_datapoints + 1) % 2);
  std::nth_element(array, array + median_idx, array + n_datapoints,
                   DataPointCompare(axis));
  // if size is 2 we want to return the first element (the smallest one),
  // since it will be placed into the first empty spot in serial_split
  return median_idx;
}

array_size sort_and_split(std::vector<DataPoint>::iterator first_data_point,
                          std::vector<DataPoint>::iterator last_data_point,
                          int axis) {
  array_size size = last_data_point - first_data_point;
  // the second part of median_idx is needed to unbalance the split towards
  // the left region (which is the one which may parallelize with the highest
  // probability).
  array_size median_idx = size / 2 - 1 * ((size + 1) % 2);
  std::nth_element(first_data_point, first_data_point + median_idx,
                   last_data_point, DataPointCompare(axis));
  // if size is 2 we want to return the first element (the smallest one),
  // since it will be placed into the first empty spot in serial_split
  return median_idx;
}

