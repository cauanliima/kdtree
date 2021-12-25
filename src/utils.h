#ifndef UTILS_H
#define UTILS_H

#include "data_point.h"
#include "tree.h"

#include <cstring>
#include <limits>

#define EMPTY_PLACEHOLDER std::numeric_limits<int>::min()

/* return the nearest number N such that N > n and N is a sum of powers of two
   Example:
    5 -> 7 = 1 + 2 + 4
    3 -> 3 = 1 + 2
*/
inline int bigger_powersum_of_two(int n) {
  int base = 1;
  int N = 0;
  while (N < n) {
    N += base;
    base *= 2;
  }
  return N;
}

inline int smaller_powersum_of_two(int n) {
  int base = 1;
  int N = 0;
  while (N < n) {
    N += base;
    base *= 2;
  }
  return N - base / 2;
}

// transform the given DataPoint array in a 1D array such that `dims` contiguous
// items constitute a data point
inline data_type *unpack_array(DataPoint *array, int size, int dims) {
  data_type *unpacked = new data_type[size * dims];
  for (int i = 0; i < size; ++i) {
    data_type *d = array[i].data();
    std::memcpy(unpacked + i * dims, d, dims * sizeof(data_type));
  }
  return unpacked;
}

// unpack an array which may contain uninitialized items
inline data_type *unpack_risky_array(DataPoint *array, int size, int dims,
                                     bool *initialized) {
  data_type *unpacked = new data_type[size * dims];
  for (int i = 0; i < size; ++i) {
    if (initialized[i]) {
      data_type *d = array[i].data();
      std::memcpy(unpacked + i * dims, d, dims * sizeof(data_type));
    } else {
      for (int j = 0; j < dims; ++j) {
        unpacked[i * dims + j] = EMPTY_PLACEHOLDER;
      }
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
inline void rearrange_branches(data_type *dest, data_type *branch1,
                               data_type *branch2, int branches_size,
                               int dims) {
  int already_added = 0;
  // number of nodes in each branch (left and right)at the current level of
  // the tree
  int nodes = 1;
  while (already_added < 2 * branches_size) {
    // we put into the three what's inside the left subtree
    std::memcpy(dest + already_added * dims, branch1,
                nodes * dims * sizeof(data_type));
    branch1 += nodes * dims;

    // we put into the three what's inside the right subtree
    std::memcpy(dest + (nodes + already_added) * dims, branch2,
                nodes * dims * sizeof(data_type));
    branch2 += nodes * dims;

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
    - dims is the number of components for each data point
    - current_level_start contains the index of the first element of tree which
        contains an element of the current node
    - current_level_nodes contains the number of elements in this level of the
        tree (each recursive call multiplies it by two)
    - start_offset contains the offset (starting from current_level_start) for
        the root node of the subtree represented by this recursive call.
*/
inline KNode<data_type> *convert_to_knodes(data_type *tree, int size, int dims,
                                           int current_level_start,
                                           int current_level_nodes,
                                           int start_offset) {
  if (tree != nullptr && current_level_start < size) {
    int next_level_start = current_level_start + current_level_nodes * dims;
    int next_level_nodes = current_level_nodes * 2;
    int next_start_offset = start_offset * 2;

    auto left = convert_to_knodes(tree, size, dims, next_level_start,
                                  next_level_nodes, next_start_offset);
    auto right = convert_to_knodes(tree, size, dims, next_level_start,
                                   next_level_nodes, next_start_offset + 1);

    return new KNode<data_type>(tree + current_level_start +
                                    start_offset * dims,
                                left, right, current_level_start == 0);
  } else {
    return nullptr;
  }
}

#endif // UTILS_H
