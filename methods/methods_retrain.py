import time

from methods.methods_common import (
    save_pretrain_embeddings,
    count_interactions,
    pretrain_files_exist,
)


class FullRetrainMethod:
    def __init__(self, cfg, loader, model):
        self.cfg = cfg
        self.loader = loader
        self.base_model = model
        self.final_model = None

    # =========================================================
    # BASIC
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

        raise TypeError(
            "FullRetrainMethod expects model to be either:\n"
            "- an instance with clone_fresh(), or\n"
            "- a model class that can be initialized."
        )

    def _maybe_save_pretrain(self):
        """
        Chỉ lưu embedding pretrain nếu:
        - cfg.save_pretrain = True
        - và file pretrain chưa tồn tại
        """
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

    def _train_model(self, model, train_dict, prefix="Retrain"):
        # Full retrain should use global epochs
        epochs = getattr(self.cfg, "epochs", 1)

        n_users = len(train_dict)
        n_interactions = count_interactions(train_dict)

        print(f"[{prefix}] users={n_users}, interactions={n_interactions}, epochs={epochs}")

        if n_interactions == 0:
            print(f"[{prefix}] Empty train data -> skip")
            return 0.0

        start = time.time()

        if hasattr(model, "fit"):
            try:
                model.fit(train_dict, epochs=epochs)
            except TypeError:
                model.fit(train_dict)
        else:
            for ep in range(epochs):
                info = model.fit_one_epoch(train_dict)
                if getattr(self.cfg, "print_loss", False) and isinstance(info, dict):
                    loss = info.get("loss", 0.0)
                    mf_loss = info.get("mf_loss", 0.0)
                    reg_loss = info.get("reg_loss", 0.0)
                    print(
                        f"[{prefix}] Epoch {ep+1}/{epochs} "
                        f"train==[{loss:.5f}={mf_loss:.5f}+{reg_loss:.5f}]"
                    )

        return time.time() - start

    def _normalize_unlearn_inputs(self, users_to_remove=None, items_to_remove=None, interactions_to_remove=None):
        users_to_remove = sorted(set(users_to_remove or []))
        items_to_remove = sorted(set(items_to_remove or []))
        interactions_to_remove = sorted(
            set((int(u), int(i)) for u, i in (interactions_to_remove or []))
        )
        return users_to_remove, items_to_remove, interactions_to_remove

    def _build_cleaned_train_dict(self, users_to_remove=None, items_to_remove=None, interactions_to_remove=None):
        users_to_remove, items_to_remove, interactions_to_remove = self._normalize_unlearn_inputs(
            users_to_remove, items_to_remove, interactions_to_remove
        )

        user_set = set(users_to_remove)
        item_set = set(items_to_remove)
        interaction_set = set(interactions_to_remove)

        train_dict = self.loader.get_full_train_data()

        cleaned = {}
        for u, items in train_dict.items():
            if u in user_set:
                continue

            new_items = []
            for i in items:
                if i in item_set:
                    continue
                if (u, i) in interaction_set:
                    continue
                new_items.append(i)

            if len(new_items) > 0:
                cleaned[u] = new_items

        return cleaned, users_to_remove, items_to_remove, interactions_to_remove

    def _collect_retrain_stats(self, cleaned):
        total_retrain_users = len(cleaned)

        item_pool = set()
        total_retrain_interactions = 0
        for _, items in cleaned.items():
            item_pool.update(items)
            total_retrain_interactions += len(items)

        total_retrain_items = len(item_pool)

        return {
            "total_retrain_users": total_retrain_users,
            "total_retrain_items": total_retrain_items,
            "total_retrain_interactions": total_retrain_interactions,
        }

    # =========================================================
    # INITIAL TRAIN
    # =========================================================
    def initial_train(self):
        print("=== INITIAL TRAIN: Full Retrain ===")

        train_dict = self.loader.get_full_train_data()
        model = self._new_model()

        t = self._train_model(model, train_dict, prefix="RETRAIN INITIAL")

        self.final_model = model
        self._maybe_save_pretrain()

        return {
            "status": "initial_train_done",
            "train_time": t,
            "total_time": t,
        }

    # =========================================================
    # UNLEARN
    # =========================================================
    def unlearn(self, users_to_remove=None, items_to_remove=None, interactions_to_remove=None):
        print("\n=== UNLEARN: Full Retrain ===")

        cleaned, users_to_remove, items_to_remove, interactions_to_remove = self._build_cleaned_train_dict(
            users_to_remove, items_to_remove, interactions_to_remove
        )

        retrain_stats = self._collect_retrain_stats(cleaned)

        print("[RETRAIN DATA SIZE]")
        print(f"  total_retrain_users        : {retrain_stats['total_retrain_users']}")
        print(f"  total_retrain_items        : {retrain_stats['total_retrain_items']}")
        print(f"  total_retrain_interactions : {retrain_stats['total_retrain_interactions']}")

        model = self._new_model()
        t = self._train_model(model, cleaned, prefix="RETRAIN UNLEARN")

        self.final_model = model
        self._maybe_save_pretrain()

        sec_per_retrain_interaction = (
            t / retrain_stats["total_retrain_interactions"]
            if retrain_stats["total_retrain_interactions"] > 0 else 0.0
        )

        return self.final_model, {
            "status": "unlearn_done",
            "affected_users": users_to_remove,
            "affected_items": items_to_remove,
            "affected_interactions": interactions_to_remove,

            "n_affected_shards": 1 if retrain_stats["total_retrain_interactions"] > 0 else 0,
            "total_retrain_users": retrain_stats["total_retrain_users"],
            "total_retrain_items": retrain_stats["total_retrain_items"],
            "total_retrain_interactions": retrain_stats["total_retrain_interactions"],
            "sec_per_retrain_interaction": sec_per_retrain_interaction,

            "affected_shard_time": t,
            "agg_train_time": 0.0,
            "retrain_time": t,
            "total_time": t,
        }

    # =========================================================
    # FINAL MODEL
    # =========================================================
    def get_final_model(self):
        return self.final_model