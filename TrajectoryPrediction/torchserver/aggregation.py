import torch
import torch.nn.functional as F
import numpy as np
from numpy import inf
from sklearn.cluster import AgglomerativeClustering, KMeans

from torch_utils import clip_tensor_norm_

from control_params import params as cp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()

def _pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.

    Arguments:
        vectors {list} -- A list of vectors.

    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)
    vectors = [v.flatten() for v in vectors]

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances

def _compute_scores(distances, i, n, f):
    """Compute scores for node i.

    Args:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
        i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.

    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)

def _multi_krum(distances, n, f, m):
    """Multi_Krum algorithm.

    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
         i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.

    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: "
                    "Got {}.".format(i, j, distances[i][j])
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]

def _geometric_median_objective(median, points, alphas):
    return sum(
        [
            alpha * _compute_euclidean_distance(median, p)
            for alpha, p in zip(alphas, points)
        ]
    )

def _clip(v: torch.Tensor, tau: float):
    v_norm = torch.norm(v)
    scale = min(1, tau / v_norm)
    return v * scale

def _mean(updates: torch.Tensor):
    return updates.mean(dim=0)

def _median(updates: torch.Tensor):
    values_upper, _ = updates.median(dim=0)
    values_lower, _ = (-updates).median(dim=0)
    res = (values_upper - values_lower) / 2
    return res

def _multiKrum(
    updates: torch.Tensor,
    num_excluded: int = 3,
    num_aggregation: int = 1,
):
    distances = _pairwise_euclidean_distances(updates)
    top_m_indices = _multi_krum(
        distances, len(updates), num_excluded, num_aggregation
    )
    values = torch.stack([updates[i] for i in top_m_indices], dim=0).mean(dim=0)
    return values

def _geoMed(
    updates: torch.Tensor,
    maxiter: int = 100,
    eps: float = 1e-6,
    ftol: float = 1e-10,
    weights=None,
):
    if weights is None:
        weights = np.ones(len(updates)) / len(updates)
    median = updates.mean(dim=0)
    num_oracle_calls = 1
    obj_val = _geometric_median_objective(median, updates, weights)
    for i in range(maxiter):
        _, prev_obj_val = median, obj_val
        weights = np.asarray(
            [
                max(
                    eps,
                    alpha
                    / max(eps, _compute_euclidean_distance(median, p).item()),
                )
                for alpha, p in zip(weights, updates)
            ],
            dtype=weights.dtype,
        )
        weights = weights / weights.sum()
        median = torch.sum(
            torch.vstack([w * beta for w, beta in zip(updates, weights)]),
            dim=0,
        )
        num_oracle_calls += 1
        obj_val = _geometric_median_objective(median, updates, weights)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break

    return median

def _autoGM(
    updates: torch.Tensor,
    lamb: float = 2.0,
    maxiter: int = 100,
    eps: float = 1e-6,
    ftol: float = 1e-10,
    weights=None,
):
    lamb = 1 * len(updates) if lamb is None else lamb
    alpha = np.ones(len(updates)) / len(updates)
    median = _geoMed(
        updates,
        maxiter=maxiter,
        eps=eps,
        ftol=ftol,
        weights=alpha
    )
    obj_val = _geometric_median_objective(median, updates, alpha)
    global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
    distance = np.zeros_like(alpha)
    for i in range(maxiter):
        prev_global_obj = global_obj
        for idx, local_model in enumerate(updates):
            distance[idx] = _compute_euclidean_distance(local_model, median)

        idxs = [x for x, _ in sorted(enumerate(distance), key=lambda x: x)]
        eta_optimal = 10000000000000000.0
        for p in range(0, len(idxs)):
            eta = (sum([distance[i] for i in idxs[: p + 1]]) + lamb) / (p + 1)
            if p < len(idxs) and eta - distance[idxs[p]] < 0:
                break
            else:
                eta_optimal = eta
        alpha = np.array([max(eta_optimal - d, 0) / lamb for d in distance])

        median = median = _geoMed(
            updates,
            maxiter=maxiter,
            eps=eps,
            ftol=ftol,
            weights=alpha
        )
        gm_sum = _geometric_median_objective(median, updates, alpha)
        global_obj = gm_sum + lamb * np.linalg.norm(alpha) ** 2 / 2
        if abs(prev_global_obj - global_obj) < ftol * global_obj:
            break
    return median

def _trimmedMean(updates: torch.Tensor, num_excluded=3):
    if len(updates) - 2 * num_excluded > 0:
        b = num_excluded
    else:
        b = num_excluded
        while len(updates) - 2 * b <= 0:
            b -= 1
        if b < 0:
            raise RuntimeError
    
    largest, _ = torch.topk(updates, b, 0)
    neg_smallest, _ = torch.topk(-updates, b, 0)
    new_stacked = torch.cat([updates, -largest, neg_smallest]).sum(0)
    new_stacked /= len(updates) - 2 * b
    return new_stacked

def _centeredClipping(
    updates: torch.Tensor,
    tau: float = 10.0,
    n_iter: int = 5,
    momentum=None,
):
    if momentum is None:
        momentum = torch.zeros_like(updates[0])
    for _ in range(n_iter):
        momentum = (
            sum(_clip(v - momentum, tau) for v in updates) / len(updates)
            + momentum
        )

    return torch.clone(momentum)

def _clustering(updates: torch.Tensor):
    num = len(updates)
    dis_max = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            dis_max[i, j] = 1 - F.cosine_similarity(
                updates[i, :], updates[j, :], dim=0
            )
            dis_max[j, i] = dis_max[i, j]
    dis_max[dis_max == -inf] = -1
    dis_max[dis_max == inf] = 1
    dis_max[np.isnan(dis_max)] = -1
    clustering = AgglomerativeClustering(
        affinity="precomputed", linkage="complete", n_clusters=2
    )
    clustering.fit(dis_max)
    flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
    values = torch.vstack(
        list(
            model
            for model, label in zip(updates, clustering.labels_)
            if label == flag
        )
    ).mean(dim=0)
    return values

def _clippedClustering(
    updates: torch.Tensor,
    agg="mean",
    signguard=False,
    max_tau=1e5,
    linkage="average",
):
    assert linkage in ["average", "single"]
    tau = max_tau
    l2norm_his = []
    l2norms = [torch.norm(update).item() for update in updates]
    l2norm_his.extend(l2norms)
    threshold = np.median(l2norm_his)
    threshold = min(threshold, tau)

    # print(threshold, l2norms)
    for idx, l2 in enumerate(l2norms):
        if l2 > threshold:
            updates[idx] = clip_tensor_norm_(updates[idx], threshold)

    num = len(updates)
    dis_max = np.zeros((num, num))
    for i in range(num):
        for j in range(i + 1, num):
            dis_max[i, j] = 1 - F.cosine_similarity(
                updates[i, :], updates[j, :], dim=0
            )
            dis_max[j, i] = dis_max[i, j]
    dis_max[dis_max == -inf] = 0
    dis_max[dis_max == inf] = 2
    dis_max[np.isnan(dis_max)] = 2
    clustering = AgglomerativeClustering(
        affinity="precomputed", linkage=linkage, n_clusters=2
    )
    clustering.fit(dis_max)

    flag = 1 if np.sum(clustering.labels_) > num // 2 else 0
    S1_idxs = list(
        [idx for idx, label in enumerate(clustering.labels_) if label == flag]
    )
    selected_idxs = S1_idxs

    if signguard:
        features = []
        num_para = len(updates[0])
        for update in updates:
            feature0 = (update > 0).sum().item() / num_para
            feature1 = (update < 0).sum().item() / num_para
            feature2 = (update == 0).sum().item() / num_para

            features.append([feature0, feature1, feature2])

        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

        flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
        S2_idxs = list(
            [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
        )

        selected_idxs = list(set(S1_idxs) & set(S2_idxs))

    benign_updates = []
    for idx in selected_idxs:
        benign_updates.append(updates[idx])

    if agg == "mean":
        values = torch.vstack(benign_updates).mean(dim=0)
    elif agg == "median":
        values = _median(torch.vstack(benign_updates))
    else:
        raise NotImplementedError(f"{agg} is not supported yet.")
    return values

def _dnc(
    updates: torch.Tensor,
    num_byzantine: int = 3,
    sub_dim: int = 1500,
    num_iters: int = 1,
    fliter_frac: float = 1.0,
):
    d = len(updates[0])

    benign_ids = []
    for i in range(num_iters):
        indices = torch.randperm(d)[: sub_dim]
        sub_updates = updates[:, indices]
        mu = sub_updates.mean(dim=0)
        centered_update = sub_updates - mu
        v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
        s = np.array(
            [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
        )

        good = s.argsort()[
            : len(updates) - int(fliter_frac * num_byzantine)
        ]
        benign_ids.extend(good)

    benign_ids = list(set(benign_ids))
    benign_updates = updates[benign_ids, :].mean(dim=0)
    return benign_updates

def _signGuard(
    updates: torch.Tensor,
    agg="mean",
    linkage="average",
):
    assert linkage in ["average", "single"]
    num = len(updates)
    l2norms = [torch.norm(update).item() for update in updates]
    M = np.median(l2norms)
    L = 0.1
    R = 3.0
    # S1 = []
    S1_idxs = []
    for idx, (l2norm, update) in enumerate(zip(l2norms, updates)):
        if l2norm >= L * M and l2norm <= R * M:
            # S1.append(update)
            S1_idxs.append(idx)

    features = []
    num_para = len(updates[0])
    for update in updates:
        feature0 = (update > 0).sum().item() / num_para
        feature1 = (update < 0).sum().item() / num_para
        feature2 = (update == 0).sum().item() / num_para

        features.append([feature0, feature1, feature2])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    print(kmeans)

    flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
    S2_idxs = list(
        [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
    )

    inter = list(set(S1_idxs) & set(S2_idxs))

    benign_updates = []
    for idx in inter:
        if l2norms[idx] > M:
            updates[idx] = clip_tensor_norm_(updates[idx], M)
        benign_updates.append(updates[idx])

    if agg == "mean":
        values = torch.vstack(benign_updates).mean(dim=0)
    elif agg == "median":
        values = _median(torch.vstack(benign_updates))
    else:
        raise NotImplementedError(f"{agg} is not supported yet.")
    return values

def aggregate(accepted_gradients: list):
    gradients = torch.from_numpy(np.array(accepted_gradients)).to(
        dtype=torch.float
    )

    aggregation_method = {
        'multi-Krum': _multiKrum,
        'GeoMed': _geoMed,
        'AutoGM': _autoGM,
        'Median': _median,
        'TrimmedMean': _trimmedMean,
        'CenteredClipping': _centeredClipping,
        'Clustering': _clustering,
        'ClippedClustering': _clippedClustering,
        'DnC': _dnc,
        'SignGuard': _signGuard,
        'Mean': _mean,
    }
    
    if cp['AGGREGATION'] in aggregation_method:
        res = aggregation_method[cp['AGGREGATION']](gradients)

    else:
        raise NotImplementedError(
            "This aggregation method has not been implemented."
        )
    
    return res.detach().numpy().tolist()