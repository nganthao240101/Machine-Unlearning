import copy
import math
import os
import pickle as pkl
import time
import numpy as np


class SISAEnsembleModel:
    """
    Final SISA model for recommender:
    - keep one trained model per shard
    - aggregate prediction scores across shard models
    """

    def __init__(self, shard_models):
        self.shard_models = dict(shard_models)

    def predict(self, user_id):
        if len(self.shard_models) == 0:
            raise ValueError("No shard models available for SISA ensemble.")

        score_list = []
        for _, model in sorted(self.shard_models.items()):
            scores = model.predict(user_id)
            score_list.append(np.asarray(scores, dtype=np.float32))

        return np.mean(np.stack(score_list, axis=0), axis=0)

    def get_state(self):
        return None

    def set_state(self, state):
        return None


class SISAMethod:
    """
    SISA for recommender project.

    Design choice in this project:
    - SISA uses interaction-based shards only
    - within each shard: cumulative slices
    - inference uses prediction aggregation across shard models

    Updated to be closer to sample SISA code:
    - checkpoint per stage
    - load previous stage checkpoint
    - train each slice with slice-specific epochs
      instead of training full epochs on every stage
    """

    def __init__(self, cfg, loader, model):
        self.cfg = cfg
        self.loader = loader

        if model is None:
            raise ValueError("SISAMethod requires a model class or model instance.")

        self.base_model = model
        self.partition_type = "interaction_based"
        self.slice_num = int(getattr(cfg, "slice_num", 3))
        self.final_model = None

        # total epochs intended for one shard's full SISA training
        self.total_epochs = int(getattr(cfg, "epochs", 1))

        self.original_shards = copy.deepcopy(loader.shards)
        self.original_shard_data = copy.deepcopy(loader.shard_data)
        self.original_train_user_dict = copy.deepcopy(loader.train_user_dict)
        self.original_exist_users = copy.deepcopy(loader.exist_users)
        self.original_n_train = loader.n_train

        self.shard_models = {}
        self.shard_slices = {}

        os.makedirs(getattr(self.cfg, "ckpt_dir", "ckpt"), exist_ok=True)

    # =========================================================
    # helpers
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
            "SISAMethod expects model to be either:\n"
            "- an instance with clone_fresh(), or\n"
            "- a model class that can be initialized."
        )

    def _ckpt_path(self, shard_id, stage_id):
        return os.path.join(
            self.cfg.ckpt_dir,
            f"sisa_shard{shard_id}_stage{stage_id}.pkl"
        )

    def _save_state(self, state, path):
        with open(path, "wb") as f:
            pkl.dump(state, f, protocol=pkl.HIGHEST_PROTOCOL)

    def _load_state(self, path):
        with open(path, "rb") as f:
            return pkl.load(f)

    def _get_slices_for_shard(self, shard_id):
        """
        loader.build_sisa_slices(shard_id) should return cumulative slices.
        Cache only for initial training; during unlearning we rebuild.
        """
        if shard_id not in self.shard_slices:
            self.shard_slices[shard_id] = self.loader.build_sisa_slices(shard_id)
        return self.shard_slices[shard_id]

    def _rebuild_slices_for_unlearned_shard(self, shard_id):
        """
        Rebuild cumulative slices from post-unlearn shard data.
        """
        return self.loader.build_sisa_slices(shard_id)

    def _reset_loader_to_original(self):
        self.loader.shards = copy.deepcopy(self.original_shards)
        self.loader.shard_data = copy.deepcopy(self.original_shard_data)
        self.loader.unlearned_shard_data = copy.deepcopy(self.original_shard_data)
        self.loader.train_user_dict = copy.deepcopy(self.original_train_user_dict)
        self.loader.exist_users = copy.deepcopy(self.original_exist_users)
        self.loader.n_train = self.original_n_train

        if hasattr(self.loader, "_invalidate_adj_cache"):
            self.loader._invalidate_adj_cache()

    def _build_final_ensemble(self):
        if len(self.shard_models) == 0:
            raise ValueError("No shard models available for SISA ensemble.")
        self.final_model = SISAEnsembleModel(self.shard_models)
        return self.final_model

    def _calc_stats(self, train_dict):
        n_users = len(train_dict)
        n_interactions = sum(len(v) for v in train_dict.values())
        item_pool = set()
        for _, items in train_dict.items():
            item_pool.update(items)
        n_items = len(item_pool)
        return n_users, n_items, n_interactions

    def _find_earliest_affected_slice(
        self,
        shard_id,
        users_to_remove=None,
        items_to_remove=None,
        interactions_to_remove=None,
        slices=None,
    ):
        users_to_remove = set(users_to_remove or [])
        items_to_remove = set(items_to_remove or [])
        interactions_to_remove = set((int(u), int(i)) for u, i in (interactions_to_remove or []))

        if slices is None:
            slices = self._get_slices_for_shard(shard_id)

        for slice_id, slice_user_dict in enumerate(slices):
            affected = False

            if users_to_remove and any(u in users_to_remove for u in slice_user_dict.keys()):
                affected = True

            if (not affected) and items_to_remove:
                for _, items in slice_user_dict.items():
                    if any(i in items_to_remove for i in items):
                        affected = True
                        break

            if (not affected) and interactions_to_remove:
                for u, items in slice_user_dict.items():
                    for i in items:
                        if (u, i) in interactions_to_remove:
                            affected = True
                            break
                    if affected:
                        break

            if affected:
                return slice_id

        return None

    def _get_slice_epoch_schedule(self, n_slices):
        """
        Closer to sample SISA code.

        Sample code distributes training budget across slices by:
            avg_epochs_per_slice = 2 * S / (S + 1) * E / S
        then:
            slice_epochs[sl] =
                int((sl+1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)

        This makes the sum of slice epochs close to total_epochs,
        and avoids training full epochs on every stage.
        """
        n_slices = max(1, int(n_slices))
        total_epochs = max(1, int(self.total_epochs))

        avg_epochs_per_slice = (2.0 * n_slices / (n_slices + 1.0)) * (float(total_epochs) / float(n_slices))

        schedule = []
        for sl in range(n_slices):
            ep = int((sl + 1) * avg_epochs_per_slice) - int(sl * avg_epochs_per_slice)
            schedule.append(max(1, ep))

        return schedule

    def _fit_model_for_epochs(self, model, train_dict, epochs):
        """
        Wrapper to support both BPR and LightGCN wrappers
        as long as they expose fit(train_user_dict, epochs=?).
        """
        epochs = max(1, int(epochs))
        return model.fit(train_dict, epochs=epochs)

    # =========================================================
    # initial train
    # =========================================================
    def initial_train(self):
        print("=== INITIAL TRAIN: SISA ===")
        print("  partition_type=interaction_based")
        print(f"  slice_num={self.slice_num}")
        print(f"  total_epochs={self.total_epochs}")

        start = time.time()
        shard_train_time = {}

        self.shard_models = {}
        self.shard_slices = {}

        for shard_id in range(len(self.loader.shards)):
            print(f"\n[SISA TRAIN] shard={shard_id}")

            shard_model = self._new_model()
            slices = self._get_slices_for_shard(shard_id)
            slice_epoch_schedule = self._get_slice_epoch_schedule(len(slices))

            shard_info = self.loader.get_shard(shard_id)
            if shard_info is not None:
                print(
                    f"  shard_users={shard_info['n_users']}, "
                    f"shard_items={shard_info['n_items']}, "
                    f"shard_interactions={shard_info['n_interactions']}"
                )

            print(f"  slice_epoch_schedule={slice_epoch_schedule}")

            shard_elapsed = 0.0

            for stage_id, stage_train_dict in enumerate(slices):
                cur_users, cur_items, cur_interactions = self._calc_stats(stage_train_dict)
                stage_epochs = slice_epoch_schedule[stage_id]

                print(
                    f"  [Shard {shard_id} | Stage {stage_id+1}/{len(slices)}] "
                    f"cumulative_users={cur_users}, "
                    f"cumulative_items={cur_items}, "
                    f"cumulative_interactions={cur_interactions}, "
                    f"stage_epochs={stage_epochs}"
                )

                if cur_interactions == 0:
                    print("    Empty stage data -> skip")
                    continue

                stage_start = time.time()
                self._fit_model_for_epochs(shard_model, stage_train_dict, stage_epochs)
                shard_elapsed += (time.time() - stage_start)

                self._save_state(shard_model.get_state(), self._ckpt_path(shard_id, stage_id))

            shard_train_time[shard_id] = shard_elapsed
            self.shard_models[shard_id] = shard_model

        agg_start = time.time()
        self._build_final_ensemble()
        agg_time = time.time() - agg_start

        total_shard_time = sum(shard_train_time.values())

        return {
            "status": "initial_train_done",
            "partition_type": "interaction_based",
            "slice_num": self.slice_num,
            "total_epochs": self.total_epochs,
            "shard_train_time": shard_train_time,
            "all_shard_train_time": total_shard_time,
            "agg_train_time": agg_time,
            "train_time": total_shard_time + agg_time,
            "total_time": time.time() - start,
        }

    # =========================================================
    # unlearn
    # =========================================================
    def unlearn(self, users_to_remove=None, items_to_remove=None, interactions_to_remove=None):
        print("\n=== UNLEARN: SISA ===")
        print("  partition_type=interaction_based")

        users_to_remove = sorted(set(users_to_remove or []))
        items_to_remove = sorted(set(items_to_remove or []))
        interactions_to_remove = sorted(set((int(u), int(i)) for u, i in (interactions_to_remove or [])))

        # self._reset_loader_to_original()

        start = time.time()

        affected_shards = sorted(set(
            self.loader.find_affected_shards(users_to_remove) +
            self.loader.find_affected_shards_by_items(items_to_remove) +
            self.loader.find_affected_shards_by_interactions(interactions_to_remove)
        ))

        print(f"[SISA UNLEARN] affected_shards={affected_shards}")

        if users_to_remove:
            self.loader.remove_unlearn_users_from_shards(users_to_remove)
        if items_to_remove:
            self.loader.remove_unlearn_items_from_shards(items_to_remove)
        if interactions_to_remove:
            self.loader.remove_unlearn_interactions_from_shards(interactions_to_remove)

        shard_train_time = {}
        total_retrain_users = 0
        total_retrain_items = 0
        total_retrain_interactions = 0

        for shard_id in affected_shards:
            shard_info = self.loader.get_unlearned_shard(shard_id)
            if shard_info is None:
                continue

            total_retrain_users += shard_info["n_users"]
            total_retrain_items += shard_info["n_items"]
            total_retrain_interactions += shard_info["n_interactions"]

            original_slices = self._get_slices_for_shard(shard_id)
            earliest_stage = self._find_earliest_affected_slice(
                shard_id,
                users_to_remove=users_to_remove,
                items_to_remove=items_to_remove,
                interactions_to_remove=interactions_to_remove,
                slices=original_slices,
            )

            if earliest_stage is None:
                continue

            print(f"[SISA RETRAIN] shard={shard_id}, from_stage={earliest_stage+1}")

            shard_model = self._new_model()

            if earliest_stage > 0:
                prev_path = self._ckpt_path(shard_id, earliest_stage - 1)
                if os.path.exists(prev_path):
                    print(f"  load checkpoint: {prev_path}")
                    state = self._load_state(prev_path)
                    shard_model.set_state(state)

            new_slices = self._rebuild_slices_for_unlearned_shard(shard_id)
            slice_epoch_schedule = self._get_slice_epoch_schedule(len(new_slices))

            print(f"  slice_epoch_schedule={slice_epoch_schedule}")

            shard_elapsed = 0.0

            for stage_id in range(earliest_stage, len(new_slices)):
                stage_train_dict = new_slices[stage_id]
                cur_users, cur_items, cur_interactions = self._calc_stats(stage_train_dict)
                stage_epochs = slice_epoch_schedule[stage_id]

                print(
                    f"  [Shard {shard_id} | Retrain Stage {stage_id+1}/{len(new_slices)}] "
                    f"cumulative_users={cur_users}, "
                    f"cumulative_items={cur_items}, "
                    f"cumulative_interactions={cur_interactions}, "
                    f"stage_epochs={stage_epochs}"
                )

                if cur_interactions == 0:
                    print("    Empty stage data -> skip")
                    continue

                stage_start = time.time()
                self._fit_model_for_epochs(shard_model, stage_train_dict, stage_epochs)
                shard_elapsed += (time.time() - stage_start)

                self._save_state(shard_model.get_state(), self._ckpt_path(shard_id, stage_id))

            shard_train_time[shard_id] = shard_elapsed
            self.shard_models[shard_id] = shard_model

        affected_shard_time = sum(shard_train_time.values())

        agg_start = time.time()
        self._build_final_ensemble()
        agg_time = time.time() - agg_start

        retrain_time = affected_shard_time + agg_time
        retrain_ratio = (
            float(total_retrain_interactions) / float(self.loader.n_train)
            if self.loader.n_train > 0 else 0.0
        )

        return self.final_model, {
            "status": "unlearn_done",
            "partition_type": "interaction_based",
            "affected_users": users_to_remove,
            "affected_items": items_to_remove,
            "affected_interactions": interactions_to_remove,
            "affected_shards": affected_shards,
            "n_affected_shards": len(affected_shards),
            "total_retrain_users": total_retrain_users,
            "total_retrain_items": total_retrain_items,
            "total_retrain_interactions": total_retrain_interactions,
            "retrain_ratio": retrain_ratio,
            "slice_num": self.slice_num,
            "total_epochs": self.total_epochs,
            "shard_train_time": shard_train_time,
            "affected_shard_time": affected_shard_time,
            "agg_train_time": agg_time,
            "retrain_time": retrain_time,
            "total_time": time.time() - start,
        }

    def get_final_model(self):
        return self.final_model