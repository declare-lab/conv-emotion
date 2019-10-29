import numpy as np


def cosine_similarity(s, g):
    similarity = np.sum(s * g, axis=1) / np.sqrt((np.sum(s * s, axis=1) * np.sum(g * g, axis=1)))

    # return np.sum(similarity)
    return similarity


def embedding_metric(samples, ground_truth, word2vec, method='average'):

    if method == 'average':
        # s, g: [n_samples, word_dim]
        s = [np.mean(sample, axis=0) for sample in samples]
        g = [np.mean(gt, axis=0) for gt in ground_truth]
        return cosine_similarity(np.array(s), np.array(g))
    elif method == 'extrema':
        s_list = []
        g_list = []
        for sample, gt in zip(samples, ground_truth):
            s_max = np.max(sample, axis=0)
            s_min = np.min(sample, axis=0)
            s_plus = np.absolute(s_min) <= s_max
            s_abs = np.max(np.absolute(sample), axis=0)
            s = s_max * s_plus + s_min * np.logical_not(s_plus)
            s_list.append(s)

            g_max = np.max(gt, axis=0)
            g_min = np.min(gt, axis=0)
            g_plus = np.absolute(g_min) <= g_max
            g_abs = np.max(np.absolute(gt), axis=0)
            g = g_max * g_plus + g_min * np.logical_not(g_plus)
            g_list.append(g)

        return cosine_similarity(np.array(s_list), np.array(g_list))
    elif method == 'greedy':
        sim_list = []
        for s, g in zip(samples, ground_truth):
            s = np.array(s)
            g = np.array(g).T
            sim = (np.matmul(s, g)
                   / np.sqrt(np.matmul(np.sum(s * s, axis=1, keepdims=True), np.sum(g * g, axis=0, keepdims=True))))
            sim = np.max(sim, axis=0)
            sim_list.append(np.mean(sim))

        # return np.sum(sim_list)
        return np.array(sim_list)
    else:
        raise NotImplementedError
