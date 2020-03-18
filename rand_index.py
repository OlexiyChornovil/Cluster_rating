import numpy as np
from scipy import sparse as sp
from sklearn.utils.fixes import comb
from scipy.special import comb as n_comb
import itertools

def rand_score(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    n_samples = labels_true.shape[0]
    n_classes = np.unique(labels_true).shape[0]
    n_clusters = np.unique(labels_pred).shape[0]

    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (n_classes == n_clusters == 1 or
            n_classes == n_clusters == 0 or
            n_classes == n_clusters == n_samples):
        return 1.0

    true_cluster_dict = {}
    for i in range(n_samples):
        if labels_true[i] not in true_cluster_dict:
            true_cluster_dict[labels_true[i]] = [i]
        else:
            true_cluster_dict[labels_true[i]].append(i)
    true_clusters = list(true_cluster_dict.values())

    pred_cluster_dict = {}
    for i in range(n_samples):
        if labels_pred[i] not in pred_cluster_dict:
            pred_cluster_dict[labels_pred[i]] = [i]
        else:
            pred_cluster_dict[labels_pred[i]].append(i)
    pred_clusters = list(pred_cluster_dict.values())

    true_pairs = []
    for i in true_clusters:
        true_pairs += list(itertools.combinations(i, 2))

    pred_pairs = []
    for i in pred_clusters:
        pred_pairs += list(itertools.combinations(i, 2))

    true_concord = []
    true_cluster_combis = list(itertools.combinations(true_clusters, 2))
    for combination in true_cluster_combis:
        true_concord += list(itertools.product(*list(combination)))

    pred_concord = []
    pred_cluster_combis = list(itertools.combinations(pred_clusters, 2))
    for combination in pred_cluster_combis:
        pred_concord += list(itertools.product(*list(combination)))

    tp = 0
    for t_pair in true_pairs:
        for p_pair in pred_pairs:
            if t_pair[0] == p_pair[0] and t_pair[1] == p_pair[1]:
                tp += 1

    tn = 0
    for t_con in true_concord:
        for p_con in pred_concord:
            if t_con[0] == p_con[0] and t_con[1] == p_con[1]:
                tn += 1

    # print(tp)
    # print(tn)
    num_comb = n_comb(n_samples, 2)
    # print(num_comb)
    return (tp + tn) / num_comb

if __name__ == '__main__':
    labels_true = np.array([1,1,0,0,0,0])
    labels_pred = np.array([0,0,0,1,0,1])
    score = rand_score(labels_true, labels_pred)
    print(score)