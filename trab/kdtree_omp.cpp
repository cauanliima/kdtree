#include "kdtree.h"

data_type *KDTreeGreenhouse::finalize() {
  grown_kdtree_size = tree_size;
  data_type *tree = unpack_optional_array(growing_tree, tree_size, n_components,
                                          EMPTY_PLACEHOLDER);
  return tree;
}
