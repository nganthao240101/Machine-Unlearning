import numpy as np


def evaluate(model, train_user_dict, test_user_dict, topk_list=[10, 20, 50]):
    recalls = np.zeros(len(topk_list))
    precisions = np.zeros(len(topk_list))
    ndcgs = np.zeros(len(topk_list))

    valid_users = 0

    for u, true_items in test_user_dict.items():
        if len(true_items) == 0:
            continue

        scores = model.predict(u)

        train_items = set(train_user_dict.get(u, []))
        if train_items:
            scores[list(train_items)] = -1e10

        ranked_items = np.argsort(scores)[::-1]
        true_set = set(true_items)
        valid_users += 1

        for idx, k in enumerate(topk_list):
            topk_items = ranked_items[:k]
            hits = [1 if item in true_set else 0 for item in topk_items]

            recall = sum(hits) / len(true_set)
            precision = sum(hits) / k

            dcg = 0.0
            for rank, item in enumerate(topk_items):
                if item in true_set:
                    dcg += 1.0 / np.log2(rank + 2)

            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_set), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            recalls[idx] += recall
            precisions[idx] += precision
            ndcgs[idx] += ndcg

    if valid_users == 0:
        print("No valid users in test set.")
        return {
            "recall": [0.0] * len(topk_list),
            "precision": [0.0] * len(topk_list),
            "ndcg": [0.0] * len(topk_list)
        }

    recalls /= valid_users
    precisions /= valid_users
    ndcgs /= valid_users

    print("\n=== TEST EVALUATION RESULTS ===")
    print("Recall@{}    = [{}]".format(topk_list, ", ".join(f"{x:.5f}" for x in recalls)))
    print("Precision@{} = [{}]".format(topk_list, ", ".join(f"{x:.5f}" for x in precisions)))
    print("NDCG@{}      = [{}]".format(topk_list, ", ".join(f"{x:.5f}" for x in ndcgs)))

    return {
        "recall": recalls.tolist(),
        "precision": precisions.tolist(),
        "ndcg": ndcgs.tolist()
    }