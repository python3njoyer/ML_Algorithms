import numpy as np


def compute_entropy(y):
    """
    returns entropy of a given set of data points at a single node in decision tree
    y: vector of 0s or 1s for negative and positive classes
    """

    # define purity variables
    p1 = np.sum(y == 1) / len(y)
    p2 = 1 - p1

    if len(y) > 0 and not ((p1 == 1) or (p1 == 0)):  # check that y is not empty or 100% one class
        entropy = (-p1 * np.log2(p1)) - (p2 * np.log2(p2))
    else:
        entropy = 0

    return entropy


def split_dataset(X, node_indices, feature):
    """
    splits data at a given node into two branches based on a certain feature
    X: matrix of training examples with 1 or 0 in each column for features
    """

    left_indices = []
    right_indices = []

    # divides into left branch for 1 (feature present) or right branch for 0 (feature not present)
    for j in node_indices:
        if X[j, feature] == 1:
            left_indices.append(j)
        elif X[j, feature] == 0:
            right_indices.append(j)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    """
    computes information gained from splitting a node on a given feature
    y: training labels
    node_indices: the indices of training examples currently stored in given node
    feature: column index where feature is found in matrix X
    """
    # split data using function defined above
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # create weights based on percentage of training examples split into either branch
    w_left = len(left_indices) / len(node_indices)
    w_right = len(right_indices) / len(node_indices)

    H_node = compute_entropy(y[node_indices])  # entropy at the root node
    H_left = compute_entropy(y[left_indices])  # entropy in left branch
    H_right = compute_entropy(y[right_indices])  # entropy in right branch

    # subtracts weighted avg of entropy in each branch from the entropy in the root node
    information_gain = H_node - (w_left * H_left + w_right * H_right)


    return information_gain


def get_best_split(X, y, node_indices):
    """
    returns the column index of the best feature to split on at a particular node
    """

    num_features = X.shape[1]

    # set column index of best feature to -1 if there is no best feature
    best_feature = -1

    max_info_gain = 0

    # calculates information gain from splitting on each feature, returns feature with maximum information gain
    for n in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, n)

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = n


    return best_feature
