import os


class Config:
    def __init__(self):
        # =========================
        # basic
        # =========================
        self.method_type = "receraser"   # retrain | sisa | receraser | all
        self.method = self.method_type
        self.model_type = "bpr"          # bpr | lightgcn

        # =========================
        # dataset / paths
        # =========================
        self.dataset_name = "ml-10m"      # ml-1m | ml-10m | yelp2018
        self.train_path = os.path.join("data", self.dataset_name, "train.txt")
        self.test_path = os.path.join("data", self.dataset_name, "test.txt")

        self.ckpt_dir = "ckpt"
        self.result_dir = "results"

        self.seed = 2024

        # =========================
        # training
        # =========================
        self.emb_dim = 64
        self.gcn_layers = 3
        self.lr = 0.001
        self.epochs = 10
        self.local_epochs = 3
        self.batch_size = 1024
        self.reg_lambda = 1e-3
        self.dropout = 0.8
        self.print_loss = True
        # self.eval_item_batch_size = 128
        # self.max_agg_batches = 20

        # =========================
        # partition
        # =========================
        self.receraser_partition_type = "interaction_based"   # user_based | item_based | interaction_based
        self.sisa_partition_type = "interaction_based"

        self.partition_type = self.receraser_partition_type
        self.partition_mode = self.partition_type
        self.receraser_split_mode = self.receraser_partition_type
        self.partition_method = self.partition_type
        self.shard_mode = self.partition_type

        self.shard_num = 10
        self.slice_num = 5

        # partition params for DataPartitioner
        self.interaction_partition_iters = 5
        self.interaction_capacity_ratio = 1.2
        self.user_partition_iters = 5
        self.user_capacity_ratio = 1.2
        self.item_partition_iters = 5
        self.item_capacity_ratio = 1.2

        # =========================
        # unlearning
        # =========================
        self.unlearn_type = "item"   # user | interaction | item
        self.unlearn_eval_runs = 5
        self.unlearn_seed = 2024

        self.unlearn_user_count = 1
        self.unlearn_interaction_count = 1
        self.unlearn_item_count = 1

        self.rec_enable_early_stop = True
        self.rec_agg_patience = 1

        # =========================
        # evaluation
        # =========================
        self.topk_list = [10, 20, 50]

        # =========================
        # RecEraser aggregation
        # =========================
        self.epoch_agg = 5
        self.agg_epochs = self.epoch_agg
        self.unlearn_agg_epochs = 1
        self.run_agg_after_unlearn = True
        self.agg_sample_ratio = 1.0   # 1.0 = full remaining data

        # =========================
        # cache
        # =========================
        self.use_partition_cache = True
        self.partition_cache_dir = "cache/partition"

        self.receraser_init_cache_dir = "cache/receraser_init"
        self.use_receraser_init_cache = True
        self.save_receraser_init_cache = True

        # =========================
        # pretrain
        # =========================
        self.pretrain_dir = os.path.join("data", self.dataset_name)
        self.save_pretrain = False

        self.user_pretrain_path = None
        self.item_pretrain_path = None
        self._update_pretrain_paths()

    def _update_pretrain_paths(self):
        if self.model_type == "bpr":
            self.user_pretrain_path = os.path.join(
                "data", self.dataset_name, "user_pretrain_bpr.pk"
            )
            self.item_pretrain_path = os.path.join(
                "data", self.dataset_name, "item_pretrain_bpr.pk"
            )
        elif self.model_type == "lightgcn":
            self.user_pretrain_path = os.path.join(
                "data", self.dataset_name, "user_pretrain_lightgcn.pk"
            )
            self.item_pretrain_path = os.path.join(
                "data", self.dataset_name, "item_pretrain_lightgcn.pk"
            )
        else:
            self.user_pretrain_path = os.path.join(
                "data", self.dataset_name, "user_pretrain.pk"
            )
            self.item_pretrain_path = os.path.join(
                "data", self.dataset_name, "item_pretrain.pk"
            )

    def sync_alias_fields(self):
        self.method = self.method_type

        # sync dataset paths in case dataset_name changed later
        self.train_path = os.path.join("data", self.dataset_name, "train.txt")
        self.test_path = os.path.join("data", self.dataset_name, "test.txt")
        self.pretrain_dir = os.path.join("data", self.dataset_name)

        if self.method_type == "receraser":
            self.partition_type = self.receraser_partition_type
        elif self.method_type == "sisa":
            self.partition_type = self.sisa_partition_type

        self.partition_mode = self.partition_type
        self.receraser_split_mode = self.receraser_partition_type
        self.partition_method = self.partition_type
        self.shard_mode = self.partition_type
        self.agg_epochs = self.epoch_agg

        self._update_pretrain_paths()