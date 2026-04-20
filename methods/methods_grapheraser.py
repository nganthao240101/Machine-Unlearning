import time

from methods.methods_common import (
    save_pretrain_embeddings,
    count_interactions,
    average_states,
)


class GraphEraserMethod:
    def __init__(self, cfg, loader, model):
        self.cfg = cfg
        self.loader = loader
        self.base_model = model

        self.final_model = None
        self.last_states = {}
        self.use_cache = getattr(cfg, "use_cache", False)

        self._ensure_shards()

    # =========================================================
    # MODEL
    # =========================================================
    def _new_model(self):
        if not isinstance(self.base_model, type) and hasattr(self.base_model, "clone_fresh"):
            return self.base_model.clone_fresh()

        if isinstance(self.base_model, type):
            try:
                return self.base_model(self.cfg, self.loader.n_users, self.loader.n_items)
            except TypeError:
                try:
                    return self.base_model(self.cfg, self.loader)
                except TypeError:
                    return self.base_model()

        raise TypeError("Invalid model format")

    def _safe_get_state(self, model):
        if hasattr(model, "get_state"):
            return model.get_state()
        raise AttributeError("Model must implement get_state() for GraphEraserMethod.")

    def _safe_set_state(self, model, state):
        if hasattr(model, "set_state"):
            model.set_state(state)
            return
        raise AttributeError("Model must implement set_state(state) for GraphEraserMethod.")

    # =========================================================
    # SHARD CHECK
    # =========================================================
    def _ensure_shards(self):
        if hasattr(self.loader, "shards") and self.loader.shards is not None:
            return

        if hasattr(self.loader, "_build_shards"):
            built = self.loader._build_shards()
            if built is not None:
                self.loader.shards = built

        if not hasattr(self.loader, "shards") or self.loader.shards is None:
            if hasattr(self.loader, "train_user_dict"):
                self.loader.shards = [{u: list(items) for u, items in self.loader.train_user_dict.items()}]
            else:
                raise ValueError("loader.shards is None")

    # =========================================================
    # TRAIN
    # =========================================================
    def _train_on_data(self, model, train_user_dict, prefix="GRAPH"):
        epochs = getattr(self.cfg, "local_epochs", self.cfg.epochs)

        n_users = len(train_user_dict)
        n_interactions = count_interactions(train_user_dict)

        print(f"[{prefix}] users={n_users}, interactions={n_interactions}, epochs={epochs}")

        if n_interactions == 0:
            print(f"[{prefix}] Empty shard -> skip")
            return 0.0

        start = time.time()

        if hasattr(model, "fit"):
            try:
                model.fit(train_user_dict, epochs=epochs)
            except TypeError:
                model.fit(train_user_dict)
        else:
            for ep in range(epochs):
                info = model.fit_one_epoch(train_user_dict)
                if getattr(self.cfg, "print_loss", False) and isinstance(info, dict):
                    loss = info.get("loss", 0.0)
                    mf_loss = info.get("mf_loss", 0.0)
                    reg_loss = info.get("reg_loss", 0.0)
                    print(
                        f"[{prefix}] Epoch {ep+1}/{epochs} "
                        f"train==[{loss:.5f}={mf_loss:.5f}+{reg_loss:.5f}]"
                    )

        return time.time() - start

    # =========================================================
    # AGGREGATE
    # =========================================================
    def _aggregate_states(self):
        if not self.last_states:
            self.final_model = None
            return None

        model = self._new_model()
        states = [self.last_states[sid] for sid in sorted(self.last_states.keys())]

        avg_state = average_states(states)
        self._safe_set_state(model, avg_state)

        self.final_model = model
        return model

    # =========================================================
    # INITIAL TRAIN
    # =========================================================
    def initial_train(self):
        print("=== INITIAL TRAIN: GRAPHERASER ===")

        self._ensure_shards()

        start = time.time()
        shard_train_time = {}

        for sid, shard in enumerate(self.loader.shards):
            print(f"\n[GRAPH TRAIN] shard={sid}")
            print(f"  users={len(shard)}")
            print(f"  interactions={count_interactions(shard)}")

            model = self._new_model()

            t = self._train_on_data(
                model,
                shard,
                prefix=f"GRAPH shard={sid}"
            )

            self.last_states[sid] = self._safe_get_state(model)
            shard_train_time[sid] = t

        agg_start = time.time()
        self._aggregate_states()
        agg_time = time.time() - agg_start

        if self.final_model is not None and getattr(self.cfg, "save_pretrain", False):
            save_pretrain_embeddings(self.final_model, self.cfg.pretrain_dir)

        return {
            "status": "initial_train_done",
            "shard_train_time": shard_train_time,
            "agg_train_time": agg_time,
            "train_time": sum(shard_train_time.values()) + agg_time,
            "total_time": time.time() - start,
        }

    # =========================================================
    # UNLEARN
    # =========================================================
    def unlearn(self, users_to_remove):
        print("\n=== UNLEARN: GRAPHERASER ===")

        users_to_remove = set(users_to_remove)
        start = time.time()

        shard_train_time = {}

        # hỗ trợ cả user_based lẫn interaction_based/random
        user_to_shards = getattr(self.loader, "user_to_shards", {})

        affected = set()
        for u in users_to_remove:
            if u in user_to_shards:
                affected.update(user_to_shards[u])

        affected = sorted(list(affected))

        print(f"[GRAPH UNLEARN] affected_shards={affected}")

        for sid in affected:
            shard = self.loader.shards[sid]

            cleaned = {
                u: items for u, items in shard.items()
                if u not in users_to_remove
            }

            print(f"\n[GRAPH RETRAIN] shard={sid}")
            print(f"  users={len(cleaned)}")
            print(f"  interactions={count_interactions(cleaned)}")

            model = self._new_model()

            t = self._train_on_data(
                model,
                cleaned,
                prefix=f"GRAPH RETRAIN shard={sid}"
            )

            self.last_states[sid] = self._safe_get_state(model)
            self.loader.shards[sid] = cleaned
            shard_train_time[sid] = t

        agg_start = time.time()
        self._aggregate_states()
        agg_time = time.time() - agg_start

        if self.final_model is not None and getattr(self.cfg, "save_pretrain", False):
            save_pretrain_embeddings(self.final_model, self.cfg.pretrain_dir)

        return self.final_model, {
            "status": "unlearn_done",
            "affected_users": sorted(list(users_to_remove)),
            "affected_shards": affected,
            "retrain_from": {sid: 0 for sid in affected},
            "shard_train_time": shard_train_time,
            "agg_train_time": agg_time,
            "retrain_time": sum(shard_train_time.values()) + agg_time,
            "total_time": time.time() - start,
        }

    def get_final_model(self):
        return self.final_model