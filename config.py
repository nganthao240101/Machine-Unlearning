class Config:
    def __init__(self):
        # =========================
        # basic
        # =========================
        self.method_type = "receraser"   # retrain | sisa | receraser | all
        self.method = self.method_type
        self.model_type = "bpr"    # bpr | lightgcn

        # =========================
        # dataset / paths
        # =========================
        self.dataset_name = "yelp2018"
        self.train_path = "data/yelp2018/train.txt"
        self.test_path = "data/yelp2018/test.txt"

        self.ckpt_dir = "ckpt"
        self.result_dir = "results"

        self.seed = 2024

        # =========================
        # training
        # =========================
        self.emb_dim = 64
        self.gcn_layers = 3
        self.lr = 0.001
        self.epochs = 5
        self.local_epochs = 3
        self.batch_size = 512
        self.reg_lambda = 1e-4
        self.dropout = 0.7
        self.print_loss = True

        # =========================
        # partition
        # =========================
        self.receraser_partition_type = "user_based"   # user_based | item_based | interaction_based
        self.sisa_partition_type = "interaction_based"

        self.partition_type = self.receraser_partition_type
        self.partition_mode = self.partition_type
        self.receraser_split_mode = self.receraser_partition_type
        self.partition_method = self.partition_type
        self.shard_mode = self.partition_type

        self.shard_num = 10
        self.slice_num = 5

        # =========================
        # unlearning
        # =========================
        self.unlearn_type = "user"   # user | interaction | item
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
        self.epoch_agg = 3
        self.agg_epochs = self.epoch_agg
        self.unlearn_agg_epochs = 2
        self.run_agg_after_unlearn = True
        self.agg_sample_ratio = 0.1

        # =========================
        # cache
        # =========================
        self.use_partition_cache = True
        self.partition_cache_dir = "cache/partition"

        self.receraser_init_cache_dir = "cache/receraser_init"
        self.use_receraser_init_cache = False
        self.save_receraser_init_cache = False

    def sync_alias_fields(self):
        self.method = self.method_type

        if self.method_type == "receraser":
            self.partition_type = self.receraser_partition_type
        elif self.method_type == "sisa":
            self.partition_type = self.sisa_partition_type

        self.partition_mode = self.partition_type
        self.receraser_split_mode = self.receraser_partition_type
        self.partition_method = self.partition_type
        self.shard_mode = self.partition_type
        self.agg_epochs = self.epoch_agg