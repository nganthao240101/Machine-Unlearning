
import copy
import os
import random
from collections import defaultdict

import numpy as np
import pickle as pkl
import scipy.sparse as sp

from data_partitioner import DataPartitioner


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

        self.train_path = cfg.train_path
        self.test_path = cfg.test_path
        self.method = getattr(cfg, "method", getattr(cfg, "method_type", "sisa"))
        self.partition_type = getattr(cfg, "partition_type", "user_based")
        self.shard_num = int(getattr(cfg, "shard_num", 3))
        self.slice_num = int(getattr(cfg, "slice_num", 3))
        self.seed = int(getattr(cfg, "seed", 2024))
        self.batch_size = int(getattr(cfg, "batch_size", 256))

        self.py_rng = random.Random(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

        self.train_user_dict = {}
        self.test_user_dict = {}

        self.user_to_shards = {}
        self.n_users = 0
        self.n_items = 0
        self.users = []
        self.items = []

        self.community_to_user = None
        self.community_to_item = None

        self.shards = None
        self.shard_data = None
        self.unlearned_shard_data = None

        # RecEraser metadata
        self.C = None
        self.C_U = None
        self.C_I = None
        self.n_C = []

        self.exist_users = []
        self.n_train = 0
        self.n_test = 0

        # global backup
        self._original_train_user_dict = None
        self._original_exist_users = None
        self._original_n_train = None

        # rec backup
        self._original_C = None
        self._original_C_U = None
        self._original_C_I = None
        self._original_shards = None
        self._original_shard_data = None
        self._original_unlearned_shard_data = None
        self._original_user_to_shards = None
        self._original_community_to_user = None
        self._original_community_to_item = None
        self._original_n_C = None

        # adjacency cache for LightGCN
        self._global_adj_cache = None
        self._local_adj_cache = {}

        self._load_data()

        if getattr(self.cfg, "use_partition_cache", False):
            if self._load_partition_cache():
                print("[Partition] Loaded from cache")
            else:
                print("[Partition] Cache not found -> building new partition")
                self.shards = self._build_shards()
                self._save_partition_cache()
        else:
            self.shards = self._build_shards()

        self._backup_original_train_state()
        self._backup_original_partition_state()

    # =========================================================
    # cache
    # =========================================================
    def _get_partition_cache_path(self):
        cache_dir = getattr(self.cfg, "partition_cache_dir", os.path.join("cache", "partition"))
        os.makedirs(cache_dir, exist_ok=True)

        dataset = getattr(self.cfg, "dataset_name", "dataset")
        partition_type = getattr(self.cfg, "partition_type", "user_based")
        shard_num = getattr(self.cfg, "shard_num", 5)
        seed = getattr(self.cfg, "seed", 2024)
        model_type = getattr(self.cfg, "model_type", "model")

        fname = f"{dataset}__{model_type}__{partition_type}__shard{shard_num}__seed{seed}.pkl"
        return os.path.join(cache_dir, fname)

    def _save_partition_cache(self):
        if self.method == "receraser":
            if self.C is None:
                return
            data = {"mode": "receraser", "C": self.C, "C_U": self.C_U, "C_I": self.C_I}
        else:
            if self.shards is None:
                return
            data = {
                "mode": self.method,
                "shards": self.shards,
                "partition_type": self.partition_type,
            }

        path = self._get_partition_cache_path()
        with open(path, "wb") as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

        print(f"[Partition] Saved to {path}")

    def _load_partition_cache(self):
        path = self._get_partition_cache_path()
        if not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                data = pkl.load(f)
        except Exception as e:
            print(f"[Partition Cache] Corrupted -> rebuild ({e})")
            return False

        mode = data.get("mode", None)

        if self.method == "receraser":
            if mode != "receraser":
                return False
            self.C = data["C"]
            self.C_U = data["C_U"]
            self.C_I = data["C_I"]
            self._rebuild_rec_metadata()
            return True

        if self.method == "sisa":
            if mode != "sisa":
                return False
            self.shards = data["shards"]
            self.partition_type = data.get("partition_type", self.partition_type)
            self.user_to_shards = self._build_user_to_shards(self.shards)

            self.community_to_user = {
                sid: sorted(list(shard.keys()))
                for sid, shard in enumerate(self.shards)
            }
            self.community_to_item = {}
            for sid, shard in enumerate(self.shards):
                item_set = set()
                for _, items in shard.items():
                    item_set.update(items)
                self.community_to_item[sid] = sorted(list(item_set))

            self.shard_data = self._generate_shard_data(self.shards)
            self.unlearned_shard_data = copy.deepcopy(self.shard_data)
            self._invalidate_adj_cache()
            return True

        if self.method == "retrain":
            if mode != "retrain":
                return False
            self.shards = data["shards"]
            self.partition_type = "full"

            self.user_to_shards = {u: [0] for u in self.train_user_dict.keys()}
            self.community_to_user = {0: sorted(self.train_user_dict.keys())}
            self.community_to_item = {0: sorted(self.items)}
            self.shard_data = {
                0: {
                    "partition_type": "full",
                    "users": sorted(self.train_user_dict.keys()),
                    "train_items": copy.deepcopy(self.train_user_dict),
                    "items": sorted(self.items),
                    "n_users": len(self.train_user_dict),
                    "n_items": len(self.items),
                    "n_interactions": sum(len(v) for v in self.train_user_dict.values()),
                }
            }
            self.unlearned_shard_data = copy.deepcopy(self.shard_data)
            self._invalidate_adj_cache()
            return True

        return False

    # =========================================================
    # basic io
    # =========================================================
    def _read_user_item_file(self, path):
        user_dict = {}
        max_u = -1
        max_i = -1
        total = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                arr = line.strip().split()
                if not arr:
                    continue

                u = int(arr[0])
                items = list(map(int, arr[1:])) if len(arr) > 1 else []

                user_dict[u] = items
                max_u = max(max_u, u)
                if items:
                    max_i = max(max_i, max(items))
                    total += len(items)

        return user_dict, max_u, max_i, total

    def _load_data(self):
        train_d, mu1, mi1, n_train = self._read_user_item_file(self.train_path)
        test_d, mu2, mi2, n_test = self._read_user_item_file(self.test_path)

        self.train_user_dict = train_d
        self.test_user_dict = test_d

        self.n_users = max(mu1, mu2) + 1
        self.n_items = max(mi1, mi2) + 1
        self.users = sorted(self.train_user_dict.keys())
        self.exist_users = sorted([u for u, items in self.train_user_dict.items() if len(items) > 0])
        self.n_train = n_train
        self.n_test = n_test

        item_pool = set()
        for _, items in self.train_user_dict.items():
            item_pool.update(items)
        self.items = sorted(item_pool)
        self._invalidate_adj_cache()

    def _backup_original_train_state(self):
        self._original_train_user_dict = copy.deepcopy(self.train_user_dict)
        self._original_exist_users = copy.deepcopy(self.exist_users)
        self._original_n_train = self.n_train

    def _backup_original_partition_state(self):
        self._original_C = copy.deepcopy(self.C)
        self._original_C_U = copy.deepcopy(self.C_U)
        self._original_C_I = copy.deepcopy(self.C_I)
        self._original_shards = copy.deepcopy(self.shards)
        self._original_shard_data = copy.deepcopy(self.shard_data)
        self._original_unlearned_shard_data = copy.deepcopy(self.unlearned_shard_data)
        self._original_user_to_shards = copy.deepcopy(self.user_to_shards)
        self._original_community_to_user = copy.deepcopy(self.community_to_user)
        self._original_community_to_item = copy.deepcopy(self.community_to_item)
        self._original_n_C = copy.deepcopy(self.n_C)

    def reset_global_train_data(self):
        if self._original_train_user_dict is not None:
            self.train_user_dict = copy.deepcopy(self._original_train_user_dict)
            self.exist_users = copy.deepcopy(self._original_exist_users)
            self.n_train = self._original_n_train
            self._invalidate_adj_cache()

    def reset_partition_state(self):
        self.C = copy.deepcopy(self._original_C)
        self.C_U = copy.deepcopy(self._original_C_U)
        self.C_I = copy.deepcopy(self._original_C_I)
        self.shards = copy.deepcopy(self._original_shards)
        self.shard_data = copy.deepcopy(self._original_shard_data)
        self.unlearned_shard_data = copy.deepcopy(self._original_unlearned_shard_data)
        self.user_to_shards = copy.deepcopy(self._original_user_to_shards)
        self.community_to_user = copy.deepcopy(self._original_community_to_user)
        self.community_to_item = copy.deepcopy(self._original_community_to_item)
        self.n_C = copy.deepcopy(self._original_n_C)
        self.reset_global_train_data()
        self._invalidate_adj_cache()

    def reset_all_train_state(self):
        self.reset_partition_state()

    # =========================================================
    # helpers
    # =========================================================
    def _flatten_interactions(self, user_dict):
        pairs = []
        for u, items in user_dict.items():
            for i in items:
                pairs.append((u, i))
        return pairs

    def _build_user_dict_from_interactions(self, pairs):
        out = defaultdict(list)
        for u, i in pairs:
            out[u].append(i)
        return dict(out)

    def _build_user_to_shards(self, shards):
        user_to_shards = {}
        for sid, shard in enumerate(shards):
            for u in shard.keys():
                user_to_shards.setdefault(u, []).append(sid)

        for u in user_to_shards:
            user_to_shards[u] = sorted(list(set(user_to_shards[u])))
        return user_to_shards

    def _generate_shard_data(self, shards):
        shard_data = {}

        for sid, shard in enumerate(shards):
            local_train_items = {}
            item_set = set()
            n_interactions = 0

            for u, items in shard.items():
                items = list(items)
                if len(items) == 0:
                    continue
                local_train_items[u] = items
                item_set.update(items)
                n_interactions += len(items)

            shard_data[sid] = {
                "partition_type": self.partition_type,
                "users": sorted(local_train_items.keys()),
                "train_items": local_train_items,
                "items": sorted(item_set),
                "n_users": len(local_train_items),
                "n_items": len(item_set),
                "n_interactions": n_interactions,
            }

        return shard_data

    # =========================================================
    # adjacency helpers for LightGCN
    # =========================================================
    def _invalidate_adj_cache(self):
        self._global_adj_cache = None
        self._local_adj_cache = {}

    def _normalize_adj_from_user_dict(self, user_dict):
        n_nodes = self.n_users + self.n_items
        rows = []
        cols = []
        data = []

        for u, items in user_dict.items():
            for i in items:
                rows.append(u)
                cols.append(self.n_users + i)
                data.append(1.0)

                rows.append(self.n_users + i)
                cols.append(u)
                data.append(1.0)

        if len(data) == 0:
            return sp.eye(n_nodes, dtype=np.float32).tocsr()

        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(rowsum + 1e-10, -0.5)
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat).tocsr()
        return norm_adj

    def get_adj_mat(self):
        if self._global_adj_cache is None:
            self._global_adj_cache = self._normalize_adj_from_user_dict(self.train_user_dict)
        return self._global_adj_cache

    def get_adj_mat_local(self, local_id):
        if self.C is None:
            raise ValueError("Local shard data C is not available.")
        if local_id < 0 or local_id >= len(self.C):
            raise ValueError(f"Invalid local_id={local_id}")
        if local_id not in self._local_adj_cache:
            self._local_adj_cache[local_id] = self._normalize_adj_from_user_dict(self.C[local_id])
        return self._local_adj_cache[local_id]

    def get_all_rec_adjs(self):
        adjs = [self.get_adj_mat_local(i) for i in range(len(self.C))]
        adjs.append(self.get_adj_mat())
        return adjs

    # =========================================================
    # SISA helpers
    # =========================================================
    def _build_sisa_random_shards(self):
        """
        Interaction-based random shards for SISA.
        Each interaction is assigned directly to a random shard.
        """
        rng = random.Random(self.seed)
        interactions = self._flatten_interactions(self.train_user_dict)
        rng.shuffle(interactions)

        shards = [dict() for _ in range(self.shard_num)]

        for (u, i) in interactions:
            sid = rng.randint(0, self.shard_num - 1)
            shards[sid].setdefault(u, []).append(i)

        return shards

    def build_sisa_slices(self, shard_id):
        """
        Build cumulative slices for SISA.
        """
        if self.method != "sisa":
            raise ValueError("build_sisa_slices() is only for method='sisa'.")

        if self.shard_data is None or shard_id not in self.shard_data:
            raise ValueError(f"Invalid shard_id={shard_id}")

        rng = random.Random(self.seed + shard_id)

        base_info = self.unlearned_shard_data if self.unlearned_shard_data is not None else self.shard_data
        shard_train_dict = base_info[shard_id]["train_items"]

        interactions = self._flatten_interactions(shard_train_dict)
        rng.shuffle(interactions)

        total = len(interactions)
        if total == 0:
            return [dict() for _ in range(self.slice_num)]

        slice_size = max(1, total // self.slice_num)

        slices = []
        for slice_id in range(self.slice_num):
            upto = total if slice_id == self.slice_num - 1 else min(total, (slice_id + 1) * slice_size)
            cumulative_part = interactions[:upto]
            slice_user_dict = self._build_user_dict_from_interactions(cumulative_part)
            slices.append(slice_user_dict)

        return slices

    # =========================================================
    # RecEraser metadata rebuild
    # =========================================================
    def _rebuild_rec_metadata(self):
        self.n_C = []
        self.user_to_shards = {}
        self.shards = [dict(cluster) for cluster in self.C]
        self.shard_data = {}
        self.community_to_user = {}
        self.community_to_item = {}

        for sid in range(len(self.C)):
            cleaned_cluster = {}
            item_pool = set()
            n_local = 0

            for u, items in self.C[sid].items():
                clean_items = list(items)
                if len(clean_items) == 0:
                    continue

                cleaned_cluster[u] = clean_items
                self.user_to_shards.setdefault(u, []).append(sid)
                item_pool.update(clean_items)
                n_local += len(clean_items)

            self.C[sid] = cleaned_cluster
            self.C_U[sid] = sorted(cleaned_cluster.keys())
            self.C_I[sid] = sorted(item_pool)
            self.n_C.append(n_local)

            self.community_to_user[sid] = sorted(self.C_U[sid])
            self.community_to_item[sid] = sorted(self.C_I[sid])

            self.shard_data[sid] = {
                "partition_type": self.partition_type,
                "users": sorted(self.C_U[sid]),
                "train_items": copy.deepcopy(self.C[sid]),
                "items": sorted(self.C_I[sid]),
                "n_users": len(self.C_U[sid]),
                "n_items": len(self.C_I[sid]),
                "n_interactions": n_local,
            }

        self.exist_users = sorted([u for u, items in self.train_user_dict.items() if len(items) > 0])
        self.n_train = sum(len(v) for v in self.train_user_dict.values())
        self.unlearned_shard_data = copy.deepcopy(self.shard_data)
        self._invalidate_adj_cache()

    # =========================================================
    # build shards
    # =========================================================
    def _build_shards(self):
        if self.method == "retrain":
            shards = [copy.deepcopy(self.train_user_dict)]
            self.partition_type = "full"

            self.user_to_shards = {u: [0] for u in self.train_user_dict.keys()}
            self.community_to_user = {0: sorted(self.train_user_dict.keys())}
            self.community_to_item = {0: sorted(self.items)}

            self.shard_data = {
                0: {
                    "partition_type": "full",
                    "users": sorted(self.train_user_dict.keys()),
                    "train_items": copy.deepcopy(self.train_user_dict),
                    "items": sorted(self.items),
                    "n_users": len(self.train_user_dict),
                    "n_items": len(self.items),
                    "n_interactions": sum(len(v) for v in self.train_user_dict.values()),
                }
            }
            self.unlearned_shard_data = copy.deepcopy(self.shard_data)
            self.shards = shards
            self._invalidate_adj_cache()
            return shards

        if self.method == "sisa":
            self.partition_type = getattr(self.cfg, "partition_type", "interaction_based")
            shards = self._build_sisa_random_shards()

            self.shards = shards
            self.user_to_shards = self._build_user_to_shards(shards)
            self.community_to_user = {
                sid: sorted(list(shard.keys()))
                for sid, shard in enumerate(shards)
            }

            self.community_to_item = {}
            for sid, shard in enumerate(shards):
                item_set = set()
                for _, items in shard.items():
                    item_set.update(items)
                self.community_to_item[sid] = sorted(list(item_set))

            self.shard_data = self._generate_shard_data(shards)
            self.unlearned_shard_data = copy.deepcopy(self.shard_data)
            self._invalidate_adj_cache()
            return shards

        if self.method == "receraser":
            partitioner = DataPartitioner(self.cfg)
            self.C, self.C_U, self.C_I = partitioner.partition(self.train_user_dict)
            self._rebuild_rec_metadata()
            return self.shards

        raise ValueError("method must be one of: retrain, sisa, receraser")

    # =========================================================
    # sampling
    # =========================================================
    def local_sample(self, local):
        if self.C is None or self.C_U is None or self.C_I is None:
            raise ValueError("local_sample() is only available for receraser flow.")

        if local < 0 or local >= len(self.C):
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        valid_users = [u for u in self.C_U[local] if len(self.C[local].get(u, [])) > 0]
        if len(valid_users) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        if self.batch_size <= len(valid_users):
            users = self.py_rng.sample(valid_users, self.batch_size)
        else:
            users = [self.py_rng.choice(valid_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.C[local].get(u, [])
            if len(pos_items) == 0:
                return []

            pos_batch = []
            while len(pos_batch) < num:
                pos_i_id = self.py_rng.choice(pos_items)
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            can_items = self.C_I[local] if local < len(self.C_I) else []

            if len(can_items) > 0:
                guard = 0
                max_guard = len(can_items) * 10 + 50

                while len(neg_items) < num and guard < max_guard:
                    guard += 1
                    neg_i_id = self.py_rng.choice(can_items)
                    if neg_i_id not in self.train_user_dict.get(u, []) and neg_i_id not in neg_items:
                        neg_items.append(neg_i_id)

            guard = 0
            max_guard = max(100, self.n_items * 2)
            while len(neg_items) < num and guard < max_guard:
                guard += 1
                neg_i_id = self.py_rng.randint(0, self.n_items - 1)
                if neg_i_id not in self.train_user_dict.get(u, []) and neg_i_id not in neg_items:
                    neg_items.append(int(neg_i_id))

            return neg_items

        final_users, pos_items, neg_items = [], [], []
        for u in users:
            pos = sample_pos_items_for_u(u, 1)
            if len(pos) == 0:
                continue

            neg = sample_neg_items_for_u(u, 1)
            if len(neg) == 0:
                continue

            final_users.append(u)
            pos_items.extend(pos)
            neg_items.extend(neg)

        return (
            np.asarray(final_users, dtype=np.int32),
            np.asarray(pos_items, dtype=np.int32),
            np.asarray(neg_items, dtype=np.int32),
        )

    def sample(self):
        valid_users = [u for u in self.exist_users if len(self.train_user_dict.get(u, [])) > 0]
        if len(valid_users) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        if self.batch_size <= len(valid_users):
            users = self.py_rng.sample(valid_users, self.batch_size)
        else:
            users = [self.py_rng.choice(valid_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict.get(u, [])
            if len(pos_items) == 0:
                return []

            pos_batch = []
            while len(pos_batch) < num:
                pos_i_id = self.py_rng.choice(pos_items)
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            guard = 0
            max_guard = max(100, self.n_items * 2)

            while len(neg_items) < num and guard < max_guard:
                guard += 1
                neg_id = self.py_rng.randint(0, self.n_items - 1)
                if neg_id not in self.train_user_dict.get(u, []) and neg_id not in neg_items:
                    neg_items.append(int(neg_id))

            return neg_items

        final_users, pos_items, neg_items = [], [], []
        for u in users:
            pos = sample_pos_items_for_u(u, 1)
            if len(pos) == 0:
                continue

            neg = sample_neg_items_for_u(u, 1)
            if len(neg) == 0:
                continue

            final_users.append(u)
            pos_items.extend(pos)
            neg_items.extend(neg)

        return (
            np.asarray(final_users, dtype=np.int32),
            np.asarray(pos_items, dtype=np.int32),
            np.asarray(neg_items, dtype=np.int32),
        )

    # =========================================================
    # info / getters
    # =========================================================
    def print_shard_summary(self):
        if self.shards is None:
            print("[Shard Summary] No shards")
            return

        print("[Shard Summary]")
        print(f"  partition_type = {self.partition_type}")
        print(f"  shard_num      = {len(self.shards)}")

        for sid, shard in enumerate(self.shards):
            n_users = len(shard)
            n_interactions = sum(len(v) for v in shard.values())
            item_set = set()
            for _, items in shard.items():
                item_set.update(items)
            n_items = len(item_set)

            print(f"  shard={sid} | users={n_users} | items={n_items} | interactions={n_interactions}")

    def get_shard(self, shard_id):
        if self.shard_data is None or shard_id not in self.shard_data:
            return None
        return self.shard_data[shard_id]

    def get_unlearned_shard(self, shard_id):
        if self.unlearned_shard_data is None or shard_id not in self.unlearned_shard_data:
            return None
        return self.unlearned_shard_data[shard_id]

    def get_full_train_data(self):
        return copy.deepcopy(self.train_user_dict)

    def get_full_test_data(self):
        return copy.deepcopy(self.test_user_dict)

    # =========================================================
    # affected shards
    # =========================================================
    def find_affected_shards(self, unlearn_users):
        u_set = set(unlearn_users or [])
        affected = []

        if self.method == "receraser" and self.C is not None:
            for sid in range(len(self.C)):
                if any(u in u_set for u in self.C[sid].keys()):
                    affected.append(sid)
            return sorted(affected)

        if self.shard_data is None:
            return []

        for sid, info in self.shard_data.items():
            if any(u in u_set for u in info["users"]):
                affected.append(sid)

        return sorted(affected)

    def find_affected_shards_by_items(self, unlearn_items):
        item_set = set(unlearn_items or [])
        affected = []

        if self.method == "receraser" and self.C is not None:
            for sid in range(len(self.C)):
                found = False
                for _, items in self.C[sid].items():
                    if any(i in item_set for i in items):
                        found = True
                        break
                if found:
                    affected.append(sid)
            return sorted(affected)

        if self.shard_data is None:
            return []

        for sid, info in self.shard_data.items():
            if any(i in item_set for i in info["items"]):
                affected.append(sid)

        return sorted(affected)

    def find_affected_shards_by_interactions(self, unlearn_interactions):
        interaction_set = set((int(u), int(i)) for u, i in (unlearn_interactions or []))
        affected = []

        if self.method == "receraser" and self.C is not None:
            for sid in range(len(self.C)):
                found = False
                for u, items in self.C[sid].items():
                    for i in items:
                        if (u, i) in interaction_set:
                            found = True
                            break
                    if found:
                        break
                if found:
                    affected.append(sid)
            return sorted(affected)

        if self.shard_data is None:
            return []

        for sid, info in self.shard_data.items():
            found = False
            for u, items in info["train_items"].items():
                for i in items:
                    if (u, i) in interaction_set:
                        found = True
                        break
                if found:
                    break
            if found:
                affected.append(sid)

        return sorted(affected)

    # =========================================================
    # remove unlearn targets
    # =========================================================
    def remove_unlearn_users_from_shards(self, unlearn_users):
        u_set = set(unlearn_users or [])

        if self.method == "receraser" and self.C is not None:
            affected_shards = self.find_affected_shards(unlearn_users)

            for sid in affected_shards:
                cleaned = {
                    u: items
                    for u, items in self.C[sid].items()
                    if u not in u_set and len(items) > 0
                }
                self.C[sid] = cleaned

            self.train_user_dict = {
                u: items
                for u, items in self.train_user_dict.items()
                if u not in u_set and len(items) > 0
            }

            self._rebuild_rec_metadata()
            return affected_shards

        if self.unlearned_shard_data is None:
            return []

        affected_shards = self.find_affected_shards(unlearn_users)

        for sid in affected_shards:
            cleaned_train_items = {
                u: items
                for u, items in self.unlearned_shard_data[sid]["train_items"].items()
                if u not in u_set and len(items) > 0
            }

            item_set = set()
            n_interactions = 0
            for _, items in cleaned_train_items.items():
                item_set.update(items)
                n_interactions += len(items)

            self.unlearned_shard_data[sid]["train_items"] = cleaned_train_items
            self.unlearned_shard_data[sid]["users"] = sorted(cleaned_train_items.keys())
            self.unlearned_shard_data[sid]["items"] = sorted(item_set)
            self.unlearned_shard_data[sid]["n_users"] = len(cleaned_train_items)
            self.unlearned_shard_data[sid]["n_items"] = len(item_set)
            self.unlearned_shard_data[sid]["n_interactions"] = n_interactions

        return affected_shards

    def remove_unlearn_items_from_shards(self, unlearn_items):
        item_set = set(unlearn_items or [])

        if self.method == "receraser" and self.C is not None:
            affected_shards = self.find_affected_shards_by_items(unlearn_items)

            for sid in affected_shards:
                cleaned = {}
                for u, items in self.C[sid].items():
                    new_items = [i for i in items if i not in item_set]
                    if len(new_items) > 0:
                        cleaned[u] = new_items
                self.C[sid] = cleaned

            new_train = {}
            for u, items in self.train_user_dict.items():
                new_items = [i for i in items if i not in item_set]
                if len(new_items) > 0:
                    new_train[u] = new_items
            self.train_user_dict = new_train

            self._rebuild_rec_metadata()
            return affected_shards

        if self.unlearned_shard_data is None:
            return []

        affected_shards = self.find_affected_shards_by_items(unlearn_items)

        for sid in affected_shards:
            cleaned_train_items = {}
            for u, items in self.unlearned_shard_data[sid]["train_items"].items():
                new_items = [i for i in items if i not in item_set]
                if len(new_items) > 0:
                    cleaned_train_items[u] = new_items

            item_pool = set()
            n_interactions = 0
            for _, items in cleaned_train_items.items():
                item_pool.update(items)
                n_interactions += len(items)

            self.unlearned_shard_data[sid]["train_items"] = cleaned_train_items
            self.unlearned_shard_data[sid]["users"] = sorted(cleaned_train_items.keys())
            self.unlearned_shard_data[sid]["items"] = sorted(item_pool)
            self.unlearned_shard_data[sid]["n_users"] = len(cleaned_train_items)
            self.unlearned_shard_data[sid]["n_items"] = len(item_pool)
            self.unlearned_shard_data[sid]["n_interactions"] = n_interactions

        return affected_shards

    def remove_unlearn_interactions_from_shards(self, unlearn_interactions):
        interaction_set = set((int(u), int(i)) for u, i in (unlearn_interactions or []))

        if self.method == "receraser" and self.C is not None:
            affected_shards = self.find_affected_shards_by_interactions(unlearn_interactions)

            for sid in affected_shards:
                cleaned = {}
                for u, items in self.C[sid].items():
                    new_items = [i for i in items if (u, i) not in interaction_set]
                    if len(new_items) > 0:
                        cleaned[u] = new_items
                self.C[sid] = cleaned

            new_train = {}
            for u, items in self.train_user_dict.items():
                new_items = [i for i in items if (u, i) not in interaction_set]
                if len(new_items) > 0:
                    new_train[u] = new_items
            self.train_user_dict = new_train

            self._rebuild_rec_metadata()
            return affected_shards

        if self.unlearned_shard_data is None:
            return []

        affected_shards = self.find_affected_shards_by_interactions(unlearn_interactions)

        for sid in affected_shards:
            cleaned_train_items = {}
            for u, items in self.unlearned_shard_data[sid]["train_items"].items():
                new_items = [i for i in items if (u, i) not in interaction_set]
                if len(new_items) > 0:
                    cleaned_train_items[u] = new_items

            item_pool = set()
            n_interactions = 0
            for _, items in cleaned_train_items.items():
                item_pool.update(items)
                n_interactions += len(items)

            self.unlearned_shard_data[sid]["train_items"] = cleaned_train_items
            self.unlearned_shard_data[sid]["users"] = sorted(cleaned_train_items.keys())
            self.unlearned_shard_data[sid]["items"] = sorted(item_pool)
            self.unlearned_shard_data[sid]["n_users"] = len(cleaned_train_items)
            self.unlearned_shard_data[sid]["n_items"] = len(item_pool)
            self.unlearned_shard_data[sid]["n_interactions"] = n_interactions

        return affected_shards
