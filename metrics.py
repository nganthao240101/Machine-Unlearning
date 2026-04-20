import numpy as np


def evaluate(model, train_user_dict, test_user_dict, topk_list):
    recalls = np.zeros(len(topk_list))
    precisions = np.zeros(len(topk_list))
    ndcgs = np.zeros(len(topk_list))

    valid_users = 0

    for u, gt_items in test_user_dict.items():
        if len(gt_items) == 0:
            continue

        # 🔥 FIX AUTO-DETECT
        try:
            all_items = list(range(model.n_items))
            scores = model.predict(u, all_items)
        except TypeError:
            scores = model.predict(u)

        scores = np.array(scores)

        seen = set(train_user_dict.get(u, []))
        if seen:
            scores[list(seen)] = -1e10

        ranked = np.argsort(scores)[::-1]
        gt = set(gt_items)
        valid_users += 1

        for idx, k in enumerate(topk_list):
            topk = ranked[:k]
            hits = [1 if x in gt else 0 for x in topk]

            recalls[idx] += sum(hits) / len(gt)
            precisions[idx] += sum(hits) / k

            dcg = 0.0
            for r, item in enumerate(topk):
                if item in gt:
                    dcg += 1.0 / np.log2(r + 2)

            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), k)))
            ndcgs[idx] += dcg / idcg if idcg > 0 else 0.0

    if valid_users == 0:
        return {
            "recall": [0.0] * len(topk_list),
            "precision": [0.0] * len(topk_list),
            "ndcg": [0.0] * len(topk_list),
        }

    recalls /= valid_users
    precisions /= valid_users
    ndcgs /= valid_users

    return {
        "recall": recalls.tolist(),
        "precision": precisions.tolist(),
        "ndcg": ndcgs.tolist(),
    }