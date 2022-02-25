import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
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


def print_node(cluster_node, node_labels):
    node_id = cluster_node.get_id()
    try:
        label = node_labels[node_id]
    except KeyError as ex:
        label = str(node_id)
    d = {'name': label}
    if not cluster_node.is_leaf():
        right_subtree = print_node(cluster_node.get_right(), node_labels)
        left_subtree = print_node(cluster_node.get_left(), node_labels)
        d['children'] = [left_subtree, right_subtree]
    return d


def create_tree(linkage_matrix, node_labels):
    '''
    Creates a JSON structure amenable for plotting with a library
    such as D3.
    '''
    cluster_node = to_tree(linkage_matrix)
    full_tree = print_node(cluster_node, node_labels)
    return full_tree


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
        t = create_tree(feature_linkage_mtx, feature_mapping)
        feature_tree_output = os.path.join(working_dir, args.features_output)
        json.dump(t, open(feature_tree_output, 'w'))
    if (args.cluster_dim == 'observations') or (args.cluster_dim == 'both'):
        observation_clustering = cluster(df.T, args.dist_metric, args.linkage)
        observation_linkage_mtx = create_linkage_matrix(observation_clustering)
        t = create_tree(observation_linkage_mtx, obs_mapping)
        observation_tree_output = os.path.join(working_dir, args.observations_output)
        json.dump(t, open(observation_tree_output, 'w'))
