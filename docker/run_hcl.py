import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import deque
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import to_tree

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input',
        required=True,
        dest = 'input_matrix',
        help='The input matrix'
    )

    parser.add_argument('-d', '--dist', \
        dest = 'dist_metric',
        required=False,
        default = 'euclidean',
        help=('The distance metric to use for clustering.')
    )

    parser.add_argument('-l', '--linkage', \
        dest = 'linkage',
        required=False,
        default = 'ward',
        help=('The linkage criterion to use for clustering.')
    )

    parser.add_argument('-c', '--cluster', \
        dest = 'cluster_dim',
        required=False,
        default = 'both',
        choices = ['both','observations','features'],
        help=('Whether to cluster observations, features, or both.')
    )

    parser.add_argument('-o', '--observations_output',
        required=True,
        help='Name of the output file for the clustered observations (if any)'
    )

    parser.add_argument('-f', '--features_output',
        required=True,
        help='Name of the output file for the clustered features (if any)'
    )

    parser.add_argument('-n', '--max-depth',
        dest = 'max_depth',
        required=False,
        default = 20,
        help='The maximum depth of the resulting tree output.'
    )

    args = parser.parse_args()
    return args

def create_linkage_matrix(clustering):
    '''
    Given the results from the clustering, create a 
    linkage matrix which can be used to create a dendrogram
    '''
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([
        clustering.children_, 
        clustering.distances_,
        counts]).astype(float)


def print_tree(nodelist, node_subset, collapsed_subtrees, mapping):
    '''
    Returns a dict which has nested formatting amenable to plotting a tree 
    structure with, e.g., the D3 javascript library.

    This implementation was previously recursive, but we often exceeded the default 
    recursion depth limit. Rather than reset the limit, we changed it to be more iterative
    with dicts

    `nodelist` is a list of ClusterNode instances returned by the `to_tree` function
    `node_subset` is a list of the nodes from the cut tree. Any nodes lower than our cut
    point have been removed by collapsing the subtree.
    `collapsed_subtrees` is a dict that maps a node ID to all the leaves of the subtree. 
    `mapping` is a dict of integers to sample names (e.g. {0: 'A', 1: 'B',...})
    where the integer refers to the column position of the original matrix
    (and hence the linkage) 

    Note that this should work regardless of the ordering of `nodelist`, but by default
    scipy returns a list ordered in a manner that should not require multiple visits 
    to the nodes.
    '''
    # Split out which nodes are leaves and which aren't.
    # According to the scipy docs, the first `n` items of `nodelist` are
    # the leaves, where `n` is the number of items we are clustering
    completed_subtrees = {}
    N = len(mapping)

    # First, take the collapsed subtrees and map the integer node IDs to
    # the actual sample name
    for k, collapsed_node_ids in collapsed_subtrees.items():
        if len(collapsed_node_ids) == 1:
            # a leaf of the original tree is trivially "collapsed"
            # the node only has the label and no children.
            completed_subtrees[k] = {'name': mapping[k]}
        else:
            # a non-leaf node with >= 1 children
            completed_subtrees[k] = {
                'name': str(k),
                'children': [
                    {'name': mapping[x]} for x in collapsed_node_ids
                ]
            }

    # a dictionary which tracks each intermediate node and the left and right immediate children
    others_dict = {}
    for node in nodelist[N:]:
        # if the current node already exists in the `completed_subtrees` dict, then
        # we don't need to track its descendents since that subtree has already been
        # processed.
        if not node.id in completed_subtrees:
            # we only care about nodes which are above the cut point.
            if node.id in node_subset:
                others_dict[node.id] = (node.left.id, node.right.id)

    unresolved_nodes = deque(others_dict)
    while unresolved_nodes:
        node_id = unresolved_nodes.popleft()
        left_id, right_id = others_dict[node_id]
        if (left_id in completed_subtrees) and (right_id in completed_subtrees):
            completed_subtrees[node_id] = {
                'name': str(node_id),
                'children': [
                    # we pop since we don't need to track those subtrees once they
                    # are part of their parent's node
                    completed_subtrees.pop(left_id),
                    completed_subtrees.pop(right_id)
                ]
            }
            # the subtree rooted at this node is fully resolved. We can drop it from
            # the dictionary tracking the unresolved nodes
            others_dict.pop(node_id)
        else:
            unresolved_nodes.append(node_id)
    return completed_subtrees


def walk(root_node, max_depth = 20, level = 0):
    '''
    Performs a limited DFS to the max depth. The full tree can be quite deep,
    which is of limited value for interpretation anyway. By limiting the depth,
    we can avoid excessively large or deep data structures which can cause
    issues, e.g. with writing/reading json

    Returns a tuple where the first item is a list of the 
    current node and all descendents (including intermediate, non-leaf nodes). 
    The second item is a mapping of the current node to a list of the leaves of
    that subtree.
    '''
    if level < max_depth:
        # if we are at a leaf, it's trivial- the only "descendent" is itself.
        if root_node.is_leaf():
            return [root_node.id], {root_node.id: [root_node.id]}
        
        # Otherwise, dive deeper through recursion. The `level` keeps us from going all the way
        # down and exceeding the default recursion depth limit.
        left_node_ids, lhs_collapsed_subtrees = walk(root_node.get_left(), level = level + 1)
        right_node_ids, rhs_collapsed_subtrees = walk(root_node.get_right(), level = level + 1)
        collapsed_subtrees = {**lhs_collapsed_subtrees, **rhs_collapsed_subtrees}
        return [
            root_node.id,
            *left_node_ids,
            *right_node_ids
        ], collapsed_subtrees
    else:
        # hit the deepest point we want to go. To avoid walking this tree twice, we 
        # get the samples in this subtree using the `pre_order` method which returns a
        # list of all the leaves of the subtree rooted at this node.
        l = root_node.pre_order(lambda x: x.id)
        return [root_node.id], {root_node.id: l}


def create_tree(linkage_matrix, node_labels, max_depth):
    '''
    Creates a JSON structure amenable for plotting with a library
    such as D3.
    '''

    # root_node is a reference to the ClusterNode instance at the root of the entire tree.
    # nodelist is a list of ClusterNode instances. Per the docs 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_tree.html#scipy.cluster.hierarchy.to_tree)
    # the first n entries of nodelist correspond to the leaves.
    root_node, nodelist = to_tree(linkage_matrix, rd=True)

    # we want to avoid issues with excessively deep trees. These can cause recursion-depth issues when writing/reading the tree in
    # JSON format. In any case, we are ultimately building this tool to work with a graphical frontend, so we need to be sensitive
    # to the visualization aspect. Hence, we will perform a cut and simply merge subtrees below the cut.
    # The `walk` function does this for us.
    # `nodes` is a list of node identifiers up to some maximum depth. It includes all nodes (leaf and non-leaf) up to 
    # and including some maximum depth in the tree.
    # `collapsed_subtrees` is a dict that maps node ids to a list of all the leaves in the subtree. It essentially flattens the subtree
    nodes, collapsed_subtrees = walk(root_node, max_depth = max_depth)

    d = print_tree(nodelist, nodes, collapsed_subtrees, node_labels)

    # dictionary returned by `print_tree` has only a single key addressed by
    # the root node's ID. We only want that part, not the whole dict
    return d[root_node.id]


def cluster(df, dist_metric, linkage):
    '''
    Perform the clustering and return the cluster object.

    The first arg is a matrix or dataframe. It is assumed that the dataframe
    is oriented in the correct manner according to sklearn/scipy, which
    clusters on the rows of the matrix.

    Thus, if the matrix is (n,m) it will cluster the `n` row-wise objects, regardless
    of whether they represent observations or features
    '''
    # Due to a constraint with scikit-learn, need to massage the args a bit.
    # For instance, even with a euclidean-calculated distance matrix, the cluster
    # method still errors since it NEEDS affinity='euclidean' if linkage='ward'

    if linkage == 'ward':
        mtx = df
        affinity = 'euclidean'
    else:
        # calculate the distance matrix. By default it's given as a vector,
        # but the `squareform` function turns it into the conventional square
        # distance matrix.
        try:
            mtx = squareform(pdist(df, dist_metric))
            if np.isnan(mtx).sum() > 0:
                raise Exception('Based on your choice of distance metric ({choice}),'
                    ' there were invalid elements encountered in the resulting "distance matrix".'
                    ' This is typically caused by a metric'
                    ' which does not make sense given your data. As an example, if '
                    ' your data has multiple rows or columns of all zeros and you use correlation,'
                    ' then you will get invalid values-- we cannot calculate the correlation'
                    ' of measurements which are all zero. If you change the distance metric, then'
                    ' it may still be possible to cluster this data.'. format(choice=dist_metric)
                )
        except Exception as ex:
            sys.stderr.write('Failed when calculating the distance matrix.'
                ' Reason: {ex}'.format(ex=ex)
            )
            sys.exit(1)
        affinity = 'precomputed'

    # Now perform the clustering. We used the pdist function above to expand
    # beyond the typical offerings for the distance arg
    try:
        clustering = AgglomerativeClustering(
            affinity = affinity,
            compute_full_tree = True, # required when distance_threshold is non-None
            linkage = linkage,
            distance_threshold=0, # full tree
            n_clusters=None # required when distance_threshold is non-None
        ).fit(mtx)
        return clustering
    except Exception as ex:
        sys.stderr.write('Failed when clustering.'
            ' Reason: {ex}'.format(ex=ex)
        )
        sys.exit(1)
    
if __name__ == '__main__':
    
    args = parse_args()
    working_dir = os.path.dirname(args.input_matrix)

    # read the matrix. This is, by our convention, (num features, num samples)
    df = pd.read_table(args.input_matrix, index_col=0)

    # Fill any NAs with zeros. Other types of "out of bounds" values (like inf)
    # we don't change as we don't want to infer too much
    df = df.fillna(0)

    obs_mapping = dict(zip(np.arange(df.shape[1]), df.columns.tolist()))
    feature_mapping = dict(zip(np.arange(df.shape[0]), df.index.tolist()))

    feature_tree_output, observation_tree_output = None, None
    if (args.cluster_dim == 'features') or (args.cluster_dim == 'both'):
        feature_clustering = cluster(df, args.dist_metric, args.linkage)
        feature_linkage_mtx = create_linkage_matrix(feature_clustering)
        t = create_tree(feature_linkage_mtx, feature_mapping, args.max_depth)
        feature_tree_output = os.path.join(working_dir, args.features_output)
        json.dump(t, open(feature_tree_output, 'w'))
    if (args.cluster_dim == 'observations') or (args.cluster_dim == 'both'):
        observation_clustering = cluster(df.T, args.dist_metric, args.linkage)
        observation_linkage_mtx = create_linkage_matrix(observation_clustering)
        t = create_tree(observation_linkage_mtx, obs_mapping, args.max_depth)
        observation_tree_output = os.path.join(working_dir, args.observations_output)
        json.dump(t, open(observation_tree_output, 'w'))
