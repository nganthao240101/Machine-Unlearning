import copy
import os
import pickle as pkl
import time

from methods.methods_common import (
    save_pretrain_embeddings,
    pretrain_files_exist,
)


class RecEraserMethod:
    """
    RecEraser flow closer to the reference code:
    - local training over every shard
    - aggregation training after locals
    - optional cache, but easy to disable
    - unlearning retrains only affected shards, then retrains agg

    Important fixes in this version:
    - every unlearning run starts from the ORIGINAL partitioned data
    - every unlearning run restores the ORIGINAL initial-trained model state
    - unlearning does NOT overwrite the saved initial state
      (so repeated runs stay independent, like paper-style experiments)
    """

    def __init__(self, cfg, loader, model):
        self.cfg = cfg
        self.loader = loader
        self._init_cache_loaded = False

        if model is None:
            raise ValueError("RecEraserMethod requires a RecEraser model class or instance.")

        if isinstance(model, type):
            self.base_model = model(cfg, loader.n_users, loader.n_items)
        else:
            self.base_model = model

        if not hasattr(self.base_model, "fit_local") or not hasattr(self.base_model, "fit_agg"):
            raise TypeError(
                "RecEraser model must implement fit_local(loader, local_id, epochs) "
                "and fit_agg(loader, epochs)."
            )

        self.partition_type = getattr(cfg, "partition_type", "user_based")
        self.final_model = self.base_model

        # snapshot ORIGINAL partition / train data right after loader construction
        self.original_C = copy.deepcopy(getattr(loader, "C", None))
        self.original_C_U = copy.deepcopy(getattr(loader, "C_U", None))
        self.original_C_I = copy.deepcopy(getattr(loader, "C_I", None))
        self.original_n_C = copy.deepcopy(getattr(loader, "n_C", []))

        self.original_train_user_dict = copy.deepcopy(loader.train_user_dict)
        self.original_exist_users = copy.deepcopy(loader.exist_users)
        self.original_n_train = loader.n_train

        # keep immutable baseline after initial_train()
        self.initial_model_state = None

    # =========================================================
    # CACHE
    # =========================================================
    def _get_init_cache_path(self):
        cache_dir = getattr(
            self.cfg,
            "receraser_init_cache_dir",
            os.path.join("cache", "receraser_init"),
        )
        os.makedirs(cache_dir, exist_ok=True)

        dataset = getattr(self.cfg, "dataset_name", "dataset")
        model_type = getattr(self.cfg, "model_type", "model")
        partition_type = getattr(self.cfg, "partition_type", "user_based")
        shard_num = getattr(self.cfg, "shard_num", 5)
        seed = getattr(self.cfg, "seed", 2024)

        fname = f"{dataset}__{model_type}__{partition_type}__shard{shard_num}__seed{seed}.pkl"
        return os.path.join(cache_dir, fname)

    def _save_init_cache(self, init_stats):
        if not getattr(self.cfg, "save_receraser_init_cache", False):
            return
        if not hasattr(self.base_model, "get_state"):
            return

        path = self._get_init_cache_path()
        data = {
            "model_state": self.base_model.get_state(),
            "init_stats": init_stats,
        }

        with open(path, "wb") as f:
            pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

        print(f"[RecEraser Init Cache] Saved to {path}")

    def _load_init_cache(self):
        if not getattr(self.cfg, "use_receraser_init_cache", False):
            return None

        path = self._get_init_cache_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path, "rb") as f:
                data = pkl.load(f)
        except Exception as e:
            print(f"[RecEraser Init Cache] Corrupted -> rebuild ({e})")
            return None

        model_state = data.get("model_state", None)
        init_stats = data.get("init_stats", None)

        if model_state is None or not hasattr(self.base_model, "set_state"):
            return None

        self.base_model.set_state(model_state)
        self.final_model = self.base_model
        self.initial_model_state = copy.deepcopy(model_state)
        self._init_cache_loaded = True

        print(f"[RecEraser Init Cache] Loaded from {path}")
        return init_stats

    # =========================================================
    # INTERNAL
    # =========================================================
    def _reset_loader_to_original_partition(self):
        # Prefer the loader's own reset path when available
        if hasattr(self.loader, "reset_all_train_state"):
            self.loader.reset_all_train_state()
            return

        if hasattr(self.loader, "reset_partition_state"):
            self.loader.reset_partition_state()
            return

        if hasattr(self.loader, "reset_global_train_data"):
            self.loader.reset_global_train_data()

        # Fallback manual restore for rec metadata
        self.loader.C = copy.deepcopy(self.original_C)
        self.loader.C_U = copy.deepcopy(self.original_C_U)
        self.loader.C_I = copy.deepcopy(self.original_C_I)
        if hasattr(self.loader, "n_C"):
            self.loader.n_C = copy.deepcopy(self.original_n_C)

        self.loader.train_user_dict = copy.deepcopy(self.original_train_user_dict)
        self.loader.exist_users = copy.deepcopy(self.original_exist_users)
        self.loader.n_train = self.original_n_train

        if hasattr(self.loader, "_rebuild_rec_metadata"):
            self.loader._rebuild_rec_metadata()

    def _restore_initial_model_state(self):
        if self.initial_model_state is None:
            return
        if not hasattr(self.base_model, "set_state"):
            return
        self.base_model.set_state(copy.deepcopy(self.initial_model_state))
        self.final_model = self.base_model

    def _maybe_save_pretrain(self):
        if self.final_model is None:
            return

        if not getattr(self.cfg, "save_pretrain", False):
            return

        save_dir = getattr(self.cfg, "pretrain_dir", None)
        if not save_dir:
            return

        if pretrain_files_exist(save_dir, require_item=True):
            print("[Pretrain] user/item pretrain already exist -> skip saving")
            return

        save_pretrain_embeddings(self.final_model, save_dir)

    def _normalize_unlearn_inputs(self, users_to_remove=None, items_to_remove=None, interactions_to_remove=None):
        users_to_remove = sorted(set(users_to_remove or []))
        items_to_remove = sorted(set(items_to_remove or []))
        interactions_to_remove = sorted(set((int(u), int(i)) for u, i in (interactions_to_remove or [])))
        return users_to_remove, items_to_remove, interactions_to_remove

    def _find_affected_shards_union(self, users_to_remove, items_to_remove, interactions_to_remove):
        affected_users = self.loader.find_affected_shards(users_to_remove) if users_to_remove else []
        affected_items = self.loader.find_affected_shards_by_items(items_to_remove) if items_to_remove else []
        affected_interactions = self.loader.find_affected_shards_by_interactions(interactions_to_remove) if interactions_to_remove else []

        affected = sorted(list(set(affected_users) | set(affected_items) | set(affected_interactions)))
        return affected, {
            "user_based": affected_users,
            "item_based": affected_items,
            "interaction_based": affected_interactions,
        }

    def _collect_retrain_shard_stats(self, affected):
        retrain_shard_stats = {}
        total_retrain_users = 0
        total_retrain_items = 0
        total_retrain_interactions = 0

        for sid in affected:
            n_users = len(self.loader.C_U[sid])
            n_items = len(self.loader.C_I[sid])
            n_interactions = self.loader.n_C[sid]

            retrain_shard_stats[sid] = {
                "users": n_users,
                "items": n_items,
                "interactions": n_interactions,
            }

            total_retrain_users += n_users
            total_retrain_items += n_items
            total_retrain_interactions += n_interactions

        return retrain_shard_stats, total_retrain_users, total_retrain_items, total_retrain_interactions

    # =========================================================
    # INITIAL TRAIN
    # =========================================================
    def initial_train(self):
        print("=== INITIAL TRAIN: RecEraser ===")
        print(f"  partition_type={self.partition_type}")

        cached_stats = self._load_init_cache()
        if cached_stats is not None:
            print("[RecEraser] Skip initial training because cached state is available")
            return cached_stats

        if hasattr(self.loader, "print_shard_summary"):
            self.loader.print_shard_summary()

        init_start = time.time()
        shard_times = {}

        print("\n[GLOBAL TRAIN SIZE]")
        print(f"  total_users        : {len(self.loader.exist_users)}")
        print(f"  total_items        : {len(self.loader.items)}")
        print(f"  total_interactions : {self.loader.n_train}")

        local_epochs = getattr(self.cfg, "local_epochs", getattr(self.cfg, "epochs", 1))
        agg_epochs = getattr(self.cfg, "epoch_agg", getattr(self.cfg, "agg_epochs", 1))

        for sid in range(len(self.loader.C)):
            print(f"\n[REC TRAIN] shard={sid}")
            print(f"  partition_type={self.partition_type}")
            print(f"  users={len(self.loader.C_U[sid])}")
            print(f"  items={len(self.loader.C_I[sid])}")
            print(f"  interactions={self.loader.n_C[sid]}")

            t0 = time.time()
            local_stats = self.base_model.fit_local(
                loader=self.loader,
                local_id=sid,
                epochs=local_epochs,
            )
            shard_times[sid] = time.time() - t0

            print(f"  shard_train_time={shard_times[sid]:.4f}s")
            if local_stats is not None:
                print(f"  local_last_stats={local_stats}")

        local_total = sum(shard_times.values())

        agg_start = time.time()
        agg_stats = self.base_model.fit_agg(
            loader=self.loader,
            epochs=agg_epochs,
        )
        agg_time = time.time() - agg_start

        print("\n[REC AGG TRAIN]")
        print(f"  agg_epochs={agg_epochs}")
        print(f"  agg_train_time={agg_time:.4f}s")
        if agg_stats is not None:
            print(f"  agg_last_stats={agg_stats}")

        self.final_model = self.base_model
        if hasattr(self.base_model, "get_state"):
            self.initial_model_state = copy.deepcopy(self.base_model.get_state())
        self._maybe_save_pretrain()

        init_stats = {
            "status": "initial_train_done",
            "partition_type": self.partition_type,
            "shard_train_time": shard_times,
            "all_shard_train_time": local_total,
            "agg_train_time": agg_time,
            "train_time": local_total + agg_time,
            "total_time": time.time() - init_start,
        }

        self._save_init_cache(init_stats)
        return init_stats

    # =========================================================
    # UNLEARN
    # =========================================================
    def unlearn(self, users_to_remove=None, items_to_remove=None, interactions_to_remove=None):
        print("\n=== UNLEARN: RecEraser ===")
        print(f"  partition_type={self.partition_type}")

        users_to_remove, items_to_remove, interactions_to_remove = self._normalize_unlearn_inputs(
            users_to_remove, items_to_remove, interactions_to_remove
        )

        # IMPORTANT: make every run independent.
        self._reset_loader_to_original_partition()
        self._restore_initial_model_state()

        start = time.time()
        shard_train_time = {}
        original_total_interactions = self.loader.n_train

        affected, affected_breakdown = self._find_affected_shards_union(
            users_to_remove, items_to_remove, interactions_to_remove
        )

        print(f"[REC UNLEARN] affected_users={users_to_remove}")
        print(f"[REC UNLEARN] affected_items={items_to_remove}")
        print(f"[REC UNLEARN] affected_interactions={interactions_to_remove}")
        print(f"[REC UNLEARN] affected_shards={affected}")

        if len(affected) == 0:
            print("[REC UNLEARN] No affected shard -> skip local retraining and aggregation.")
            return self.final_model, {
                "status": "unlearn_done",
                "partition_type": self.partition_type,
                "affected_users": users_to_remove,
                "affected_items": items_to_remove,
                "affected_interactions": interactions_to_remove,
                "affected_shards": affected,
                "affected_from_users": affected_breakdown["user_based"],
                "affected_from_items": affected_breakdown["item_based"],
                "affected_from_interactions": affected_breakdown["interaction_based"],
                "retrain_shard_stats": {},
                "n_affected_shards": 0,
                "total_retrain_users": 0,
                "total_retrain_items": 0,
                "total_retrain_interactions": 0,
                "retrain_ratio": 0.0,
                "sec_per_retrain_interaction": 0.0,
                "shard_train_time": {},
                "affected_shard_time": 0.0,
                "agg_train_time": 0.0,
                "retrain_time": 0.0,
                "total_time": time.time() - start,
            }

        if users_to_remove:
            self.loader.remove_unlearn_users_from_shards(users_to_remove)
        if items_to_remove:
            self.loader.remove_unlearn_items_from_shards(items_to_remove)
        if interactions_to_remove:
            self.loader.remove_unlearn_interactions_from_shards(interactions_to_remove)

        retrain_shard_stats, total_retrain_users, total_retrain_items, total_retrain_interactions = \
            self._collect_retrain_shard_stats(affected)

        print("\n[REC RETRAIN DATA SIZE]")
        print(f"  n_affected_shards          : {len(affected)}")
        print(f"  total_retrain_users        : {total_retrain_users}")
        print(f"  total_retrain_items        : {total_retrain_items}")
        print(f"  total_retrain_interactions : {total_retrain_interactions}")

        local_epochs = getattr(self.cfg, "local_epochs", getattr(self.cfg, "epochs", 1))
        agg_epochs = getattr(
            self.cfg,
            "unlearn_agg_epochs",
            getattr(self.cfg, "epoch_agg", getattr(self.cfg, "agg_epochs", 1)),
        )

        for sid in affected:
            print(f"\n[REC RETRAIN] shard={sid}")
            print(f"  partition_type={self.partition_type}")
            print(f"  users={len(self.loader.C_U[sid])}")
            print(f"  items={len(self.loader.C_I[sid])}")
            print(f"  interactions={self.loader.n_C[sid]}")

            t0 = time.time()
            local_stats = self.base_model.fit_local(
                loader=self.loader,
                local_id=sid,
                epochs=local_epochs,
            )
            shard_train_time[sid] = time.time() - t0

            print(f"  shard_retrain_time={shard_train_time[sid]:.4f}s")
            if local_stats is not None:
                print(f"  local_last_stats={local_stats}")

        affected_shard_time = sum(shard_train_time.values())

        agg_time = 0.0
        if getattr(self.cfg, "run_agg_after_unlearn", True):
            agg_start = time.time()
            agg_stats = self.base_model.fit_agg(
                loader=self.loader,
                epochs=agg_epochs,
            )
            agg_time = time.time() - agg_start

            print("\n[REC AGG RETRAIN]")
            print(f"  agg_epochs={agg_epochs}")
            print(f"  agg_train_time={agg_time:.4f}s")
            if agg_stats is not None:
                print(f"  agg_last_stats={agg_stats}")
        else:
            print("[REC UNLEARN] Skip aggregation after unlearning")

        self.final_model = self.base_model
        # DO NOT overwrite self.initial_model_state here.
        # Keeping the original initial state is critical for independent runs.
        self._maybe_save_pretrain()

        retrain_time = affected_shard_time + agg_time
        retrain_ratio = (
            float(total_retrain_interactions) / float(original_total_interactions)
            if original_total_interactions > 0 else 0.0
        )
        sec_per_retrain_interaction = (
            retrain_time / total_retrain_interactions
            if total_retrain_interactions > 0 else 0.0
        )

        return self.final_model, {
            "status": "unlearn_done",
            "partition_type": self.partition_type,
            "affected_users": users_to_remove,
            "affected_items": items_to_remove,
            "affected_interactions": interactions_to_remove,
            "affected_shards": affected,
            "affected_from_users": affected_breakdown["user_based"],
            "affected_from_items": affected_breakdown["item_based"],
            "affected_from_interactions": affected_breakdown["interaction_based"],
            "retrain_shard_stats": retrain_shard_stats,
            "n_affected_shards": len(affected),
            "total_retrain_users": total_retrain_users,
            "total_retrain_items": total_retrain_items,
            "total_retrain_interactions": total_retrain_interactions,
            "retrain_ratio": retrain_ratio,
            "sec_per_retrain_interaction": sec_per_retrain_interaction,
            "shard_train_time": shard_train_time,
            "affected_shard_time": affected_shard_time,
            "agg_train_time": agg_time,
            "retrain_time": retrain_time,
            "total_time": time.time() - start,
        }

    def get_final_model(self):
        return self.final_model
