import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np


UserDict = Dict[int, List[int]]
ShardList = List[UserDict]


class DataPartitioner:
    """
    Supported partition types:
        - interaction_based
        - user_based
        - item_based
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.partition_type = getattr(cfg, "partition_type", "user_based")
        self.shard_num = int(getattr(cfg, "shard_num", 5))
        self.seed = int(getattr(cfg, "seed", 2024))

        self.interaction_partition_iters = int(getattr(cfg, "interaction_partition_iters", 5))
        self.interaction_capacity_ratio = float(getattr(cfg, "interaction_capacity_ratio", 1.2))

        self.user_partition_iters = int(getattr(cfg, "user_partition_iters", 5))
        self.user_capacity_ratio = float(getattr(cfg, "user_capacity_ratio", 1.2))

        self.item_partition_iters = int(getattr(cfg, "item_partition_iters", 5))
        self.item_capacity_ratio = float(getattr(cfg, "item_capacity_ratio", 1.2))

        self.user_pretrain_path = getattr(
            cfg,
            "user_pretrain_path",
            os.path.join("data", getattr(cfg, "dataset_name", "ml-1m"), "user_pretrain.pk"),
        )
        self.item_pretrain_path = getattr(
            cfg,
            "item_pretrain_path",
            os.path.join("data", getattr(cfg, "dataset_name", "ml-1m"), "item_pretrain.pk"),
        )

        random.seed(self.seed)
        np.random.seed(self.seed)

    # =========================================================
    # helpers
    # =========================================================
    def _build_users_items_from_clusters(self, clusters: ShardList):
        users = [[] for _ in range(len(clusters))]
        items = [[] for _ in range(len(clusters))]

        for sid, shard in enumerate(clusters):
            users[sid] = sorted(shard.keys())
            item_pool = set()
            for _, shard_items in shard.items():
                item_pool.update(shard_items)
            items[sid] = sorted(item_pool)

        return users, items

    def _flatten_interactions(self, user_dict: UserDict):
        pairs = []
        for u, items in user_dict.items():
            for i in items:
                pairs.append((u, i))
        return pairs

    def _invert_user_dict_to_item_dict(self, user_dict: UserDict):
        item_dict = {}
        for u, items in user_dict.items():
            for i in items:
                item_dict.setdefault(i, []).append(u)
        return item_dict

    def _load_user_pretrain_embeddings(self):
        if not os.path.exists(self.user_pretrain_path):
            raise FileNotFoundError(f"user_pretrain not found: {self.user_pretrain_path}")
        with open(self.user_pretrain_path, "rb") as f:
            emb = pickle.load(f)

        if isinstance(emb, dict):
            return emb

        emb = np.asarray(emb)
        return {int(i): emb[i] for i in range(emb.shape[0])}

    def _load_item_pretrain_embeddings(self):
        if not os.path.exists(self.item_pretrain_path):
            raise FileNotFoundError(f"item_pretrain not found: {self.item_pretrain_path}")
        with open(self.item_pretrain_path, "rb") as f:
            emb = pickle.load(f)

        if isinstance(emb, dict):
            return emb

        emb = np.asarray(emb)
        return {int(i): emb[i] for i in range(emb.shape[0])}

    def _to_vec(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = np.array([float(x)], dtype=np.float64)

        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        return x

    def _sq_dist(self, a, b):
        a = self._to_vec(a)
        b = self._to_vec(b)
        return float(np.sum((a - b) ** 2))

    def _mean_vecs(self, vecs, fallback):
        if len(vecs) == 0:
            return fallback
        return np.mean(np.stack([self._to_vec(v) for v in vecs], axis=0), axis=0)

    # =========================================================
    # fallback partitions
    # =========================================================
    def _balanced_user_fallback(self, user_dict: UserDict):
        clusters = [dict() for _ in range(self.shard_num)]
        loads = [0 for _ in range(self.shard_num)]

        users_sorted = sorted(user_dict.keys(), key=lambda u: len(user_dict[u]), reverse=True)
        for u in users_sorted:
            sid = int(np.argmin(loads))
            clusters[sid][u] = list(user_dict[u])
            loads[sid] += len(user_dict[u])

        users, items = self._build_users_items_from_clusters(clusters)
        return clusters, users, items

    def _balanced_item_fallback(self, user_dict: UserDict):
        item_dict = self._invert_user_dict_to_item_dict(user_dict)
        item_to_shard = {}
        loads = [0 for _ in range(self.shard_num)]

        items_sorted = sorted(item_dict.keys(), key=lambda i: len(item_dict[i]), reverse=True)
        for i in items_sorted:
            sid = int(np.argmin(loads))
            item_to_shard[i] = sid
            loads[sid] += len(item_dict[i])

        clusters = [dict() for _ in range(self.shard_num)]
        for u, items in user_dict.items():
            for i in items:
                sid = item_to_shard[i]
                clusters[sid].setdefault(u, []).append(i)

        users, items = self._build_users_items_from_clusters(clusters)
        return clusters, users, items

    def _balanced_interaction_fallback(self, user_dict: UserDict):
        pairs = self._flatten_interactions(user_dict)
        clusters = [dict() for _ in range(self.shard_num)]
        loads = [0 for _ in range(self.shard_num)]

        for u, i in pairs:
            sid = int(np.argmin(loads))
            clusters[sid].setdefault(u, []).append(i)
            loads[sid] += 1

        users, items = self._build_users_items_from_clusters(clusters)
        return clusters, users, items

    # =========================================================
    # user-based
    # =========================================================
    def user_based_partition(self, user_dict: UserDict):
        try:
            uidW = self._load_user_pretrain_embeddings()
        except FileNotFoundError as e:
            print(f"[DataPartitioner] {e}")
            print("[DataPartitioner] fallback -> balanced user partition")
            return self._balanced_user_fallback(user_dict)

        data = {u: list(items) for u, items in user_dict.items()}
        all_users = list(data.keys())

        valid_users = [u for u in all_users if u in uidW]
        missing_users = [u for u in all_users if u not in uidW]

        if len(valid_users) < self.shard_num:
            print("[DataPartitioner] not enough valid user embeddings -> fallback")
            return self._balanced_user_fallback(user_dict)

        centroids = random.sample(valid_users, self.shard_num)
        centroid_embs = [self._to_vec(uidW[u]) for u in centroids]

        max_users_per_shard = int(np.ceil(
            self.user_capacity_ratio * len(valid_users) / self.shard_num
        ))

        last_clusters = None
        for it in range(self.user_partition_iters):
            clusters = [dict() for _ in range(self.shard_num)]
            used = set()

            scored_pairs = []
            for u in valid_users:
                user_emb = uidW[u]
                for sid in range(self.shard_num):
                    score = -self._sq_dist(user_emb, centroid_embs[sid])
                    scored_pairs.append(((u, sid), score))

            scored_pairs.sort(key=lambda x: x[1], reverse=True)

            for (u, sid), _ in scored_pairs:
                if u in used:
                    continue
                if len(clusters[sid]) >= max_users_per_shard:
                    continue
                clusters[sid][u] = data[u]
                used.add(u)

            loads = [sum(len(v) for v in clusters[sid].values()) for sid in range(self.shard_num)]
            for u in valid_users:
                if u in used:
                    continue
                sid = int(np.argmin(loads))
                clusters[sid][u] = data[u]
                loads[sid] += len(data[u])
                used.add(u)

            new_centroid_embs = []
            for sid in range(self.shard_num):
                emb_list = [uidW[u] for u in clusters[sid].keys() if u in uidW]
                new_centroid_embs.append(self._mean_vecs(emb_list, centroid_embs[sid]))

            centroid_embs = new_centroid_embs
            last_clusters = clusters

            shard_sizes = [len(clusters[s]) for s in range(self.shard_num)]
            interaction_sizes = [sum(len(v) for v in clusters[s].values()) for s in range(self.shard_num)]
            print(f"[user_based_partition] iter={it} user_sizes={shard_sizes} interaction_sizes={interaction_sizes}")

        if missing_users:
            loads = [sum(len(v) for v in last_clusters[sid].values()) for sid in range(self.shard_num)]
            for u in sorted(missing_users, key=lambda x: len(data[x]), reverse=True):
                sid = int(np.argmin(loads))
                last_clusters[sid][u] = data[u]
                loads[sid] += len(data[u])

        users, items = self._build_users_items_from_clusters(last_clusters)
        return last_clusters, users, items

    # =========================================================
    # item-based
    # =========================================================
    def item_based_partition(self, user_dict: UserDict):
        try:
            iidW = self._load_item_pretrain_embeddings()
        except FileNotFoundError as e:
            print(f"[DataPartitioner] {e}")
            print("[DataPartitioner] fallback -> balanced item partition")
            return self._balanced_item_fallback(user_dict)

        item_dict = self._invert_user_dict_to_item_dict(user_dict)
        all_items = list(item_dict.keys())

        valid_items = [i for i in all_items if i in iidW]
        missing_items = [i for i in all_items if i not in iidW]

        if len(valid_items) < self.shard_num:
            print("[DataPartitioner] not enough valid item embeddings -> fallback")
            return self._balanced_item_fallback(user_dict)

        centroids = random.sample(valid_items, self.shard_num)
        centroid_embs = [self._to_vec(iidW[i]) for i in centroids]

        max_items_per_shard = int(np.ceil(
            self.item_capacity_ratio * len(valid_items) / self.shard_num
        ))

        last_item_clusters = None
        for it in range(self.item_partition_iters):
            item_clusters = [set() for _ in range(self.shard_num)]
            used = set()

            scored_pairs = []
            for i in valid_items:
                item_emb = iidW[i]
                for sid in range(self.shard_num):
                    score = -self._sq_dist(item_emb, centroid_embs[sid])
                    scored_pairs.append(((i, sid), score))

            scored_pairs.sort(key=lambda x: x[1], reverse=True)

            for (i, sid), _ in scored_pairs:
                if i in used:
                    continue
                if len(item_clusters[sid]) >= max_items_per_shard:
                    continue
                item_clusters[sid].add(i)
                used.add(i)

            loads = [sum(len(item_dict[i]) for i in item_clusters[sid]) for sid in range(self.shard_num)]
            for i in valid_items:
                if i in used:
                    continue
                sid = int(np.argmin(loads))
                item_clusters[sid].add(i)
                loads[sid] += len(item_dict[i])
                used.add(i)

            new_centroid_embs = []
            for sid in range(self.shard_num):
                emb_list = [iidW[i] for i in item_clusters[sid] if i in iidW]
                new_centroid_embs.append(self._mean_vecs(emb_list, centroid_embs[sid]))

            centroid_embs = new_centroid_embs
            last_item_clusters = item_clusters

            item_sizes = [len(item_clusters[s]) for s in range(self.shard_num)]
            interaction_sizes = [sum(len(item_dict[i]) for i in item_clusters[s]) for s in range(self.shard_num)]
            print(f"[item_based_partition] iter={it} item_sizes={item_sizes} interaction_sizes={interaction_sizes}")

        if missing_items:
            loads = [sum(len(item_dict[i]) for i in last_item_clusters[sid]) for sid in range(self.shard_num)]
            for i in sorted(missing_items, key=lambda x: len(item_dict[x]), reverse=True):
                sid = int(np.argmin(loads))
                last_item_clusters[sid].add(i)
                loads[sid] += len(item_dict[i])

        item_to_shard = {}
        for sid in range(self.shard_num):
            for i in last_item_clusters[sid]:
                item_to_shard[i] = sid

        clusters = [dict() for _ in range(self.shard_num)]
        for u, items in user_dict.items():
            for i in items:
                sid = item_to_shard[i]
                clusters[sid].setdefault(u, []).append(i)

        users, items = self._build_users_items_from_clusters(clusters)
        return clusters, users, items

    # =========================================================
    # interaction-based
    # =========================================================
    def interaction_based_partition(self, user_dict: UserDict):
        try:
            uidW = self._load_user_pretrain_embeddings()
            iidW = self._load_item_pretrain_embeddings()
        except FileNotFoundError as e:
            print(f"[DataPartitioner] {e}")
            print("[DataPartitioner] fallback -> balanced interaction partition")
            return self._balanced_interaction_fallback(user_dict)

        pairs = self._flatten_interactions(user_dict)
        valid_pairs = [(u, i) for (u, i) in pairs if u in uidW and i in iidW]
        missing_pairs = [(u, i) for (u, i) in pairs if u not in uidW or i not in iidW]

        if len(valid_pairs) < self.shard_num:
            print("[DataPartitioner] not enough valid interactions -> fallback")
            return self._balanced_interaction_fallback(user_dict)

        centroids = random.sample(valid_pairs, self.shard_num)
        centroid_embs = [[self._to_vec(uidW[u]), self._to_vec(iidW[i])] for (u, i) in centroids]

        max_interactions_per_shard = int(np.ceil(
            self.interaction_capacity_ratio * len(valid_pairs) / self.shard_num
        ))

        last_clusters = None
        for it in range(self.interaction_partition_iters):
            clusters = [dict() for _ in range(self.shard_num)]
            counts = [0 for _ in range(self.shard_num)]
            used_idx = set()

            scored_pairs = []
            for idx, (u, i) in enumerate(valid_pairs):
                for sid in range(self.shard_num):
                    score = -(
                        self._sq_dist(uidW[u], centroid_embs[sid][0]) +
                        self._sq_dist(iidW[i], centroid_embs[sid][1])
                    )
                    scored_pairs.append(((idx, sid), score))

            scored_pairs.sort(key=lambda x: x[1], reverse=True)

            for (idx, sid), _ in scored_pairs:
                if idx in used_idx:
                    continue
                if counts[sid] >= max_interactions_per_shard:
                    continue

                u, i = valid_pairs[idx]
                clusters[sid].setdefault(u, []).append(i)
                counts[sid] += 1
                used_idx.add(idx)

            for idx, (u, i) in enumerate(valid_pairs):
                if idx in used_idx:
                    continue
                sid = int(np.argmin(counts))
                clusters[sid].setdefault(u, []).append(i)
                counts[sid] += 1
                used_idx.add(idx)

            new_centroid_embs = []
            for sid in range(self.shard_num):
                user_embs = []
                item_embs = []
                for u, items in clusters[sid].items():
                    for i in items:
                        if u in uidW and i in iidW:
                            user_embs.append(uidW[u])
                            item_embs.append(iidW[i])

                if len(user_embs) == 0:
                    new_centroid_embs.append(centroid_embs[sid])
                else:
                    new_centroid_embs.append([
                        self._mean_vecs(user_embs, centroid_embs[sid][0]),
                        self._mean_vecs(item_embs, centroid_embs[sid][1]),
                    ])

            centroid_embs = new_centroid_embs
            last_clusters = clusters

            print(f"[interaction_based_partition] iter={it} interaction_sizes={counts}")

        if missing_pairs:
            loads = [sum(len(v) for v in last_clusters[sid].values()) for sid in range(self.shard_num)]
            for u, i in missing_pairs:
                sid = int(np.argmin(loads))
                last_clusters[sid].setdefault(u, []).append(i)
                loads[sid] += 1

        users, items = self._build_users_items_from_clusters(last_clusters)
        return last_clusters, users, items

    # =========================================================
    # main
    # =========================================================
    def partition(self, user_dict: UserDict):
        if self.partition_type == "interaction_based":
            return self.interaction_based_partition(user_dict)
        if self.partition_type == "user_based":
            return self.user_based_partition(user_dict)
        if self.partition_type == "item_based":
            return self.item_based_partition(user_dict)

        raise ValueError(
            "partition_type must be one of: interaction_based, user_based, item_based"
        )