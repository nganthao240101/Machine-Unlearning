# params.py
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run unlearning experiments")

    # =========================================================
    # DATA
    # =========================================================
    parser.add_argument("--dataset_name", type=str, default="ml-1m")
    parser.add_argument("--train_path", type=str, default="data/train.txt")
    parser.add_argument("--test_path", type=str, default="data/test.txt")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")

    # =========================================================
    # MODEL / METHOD
    # =========================================================
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpr",
        choices=["bpr", "wmf", "lightgcn"],
        help="Model type"
    )

    parser.add_argument(
        "--method_type",
        type=str,
        default="all",
        choices=["retrain", "sisa", "receraser", "grapheraser", "all"],
        help="Method type"
    )

    # =========================================================
    # TRAINING
    # =========================================================
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--reg_lambda", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--print_loss", type=str, default="true")
    parser.add_argument("--gcn_layers", type=int, default=2)

    # =========================================================
    # SHARD / SLICE
    # =========================================================
    parser.add_argument("--shard_num", type=int, default=3)
    parser.add_argument("--slice_num", type=int, default=3)

    # mode chung duy nhất cho toàn pipeline
    parser.add_argument(
        "--partition_mode",
        type=str,
        default="random",
        choices=["random", "user_grouped"],
        help="Unified partition mode for all methods"
    )

    # các field cũ giữ lại để tương thích, sẽ được sync lại trong main.py
    parser.add_argument("--shard_mode", type=str, default="user")
    parser.add_argument(
        "--slice_mode",
        type=str,
        default="random",
        choices=["random", "user_grouped"],
        help="SISA slice mode (legacy-compatible field)"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Legacy SISA field; kept for compatibility."
    )

    # =========================================================
    # LOCAL METHOD
    # =========================================================
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--agg_dummy_time", type=float, default=0.0)

    # =========================================================
    # UNLEARN / EVAL
    # =========================================================
    parser.add_argument(
        "--unlearn_users",
        type=int,
        nargs="+",
        default=[0, 3],
        help="Users to unlearn"
    )

    parser.add_argument(
        "--topk_list",
        type=int,
        nargs="+",
        default=[10, 20, 50]
    )

    # =========================================================
    # RecEraser
    # =========================================================
    parser.add_argument(
        "--receraser_split_mode",
        type=str,
        default="random",
        choices=["user-group", "random"]
    )

    # =========================================================
    # GraphEraser
    # =========================================================
    parser.add_argument(
        "--partition_method",
        type=str,
        default="random",
        choices=["random", "user_grouped", "kmeans", "user_group"]
    )

    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)

    # =========================================================
    # PARTITION_2_LIKE
    # =========================================================
    parser.add_argument("--partition2_iters", type=int, default=5)
    parser.add_argument("--partition2_capacity_ratio", type=float, default=1.2)
    parser.add_argument("--user_pretrain_path", type=str, default="ckpt/pretrain/user_pretrain.pk")

    # =========================================================
    # PRETRAIN / CACHE
    # =========================================================
    parser.add_argument("--pretrain_dir", type=str, default="ckpt/pretrain")
    parser.add_argument("--save_pretrain", type=str, default="false")
    parser.add_argument("--use_pretrain", type=str, default="false")

    parser.add_argument("--use_cache", type=str, default="false")
    parser.add_argument("--save_cache", type=str, default="false")

    parser.add_argument("--receraser_cache_dir", type=str, default="ckpt/receraser_cache")
    parser.add_argument("--grapheraser_cache_dir", type=str, default="ckpt/grapheraser_cache")

    args = parser.parse_args()
    return normalize_args(args)


# =========================================================
# UTILS
# =========================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ["true", "1", "yes", "y"]:
        return True
    if v in ["false", "0", "no", "n"]:
        return False
    raise ValueError(f"Cannot convert to bool: {v}")


def normalize_args(args):
    args.print_loss = str2bool(args.print_loss)
    args.save_pretrain = str2bool(args.save_pretrain)
    args.use_pretrain = str2bool(args.use_pretrain)
    args.use_cache = str2bool(args.use_cache)
    args.save_cache = str2bool(args.save_cache)

    args = apply_new_architecture_fields(args)
    return args


# =========================================================
# ARCHITECTURE MAPPING
# =========================================================
def apply_new_architecture_fields(args):
    # đồng bộ field chính
    args.method = args.method_type

    # normalize alias
    if args.partition_mode in ["user-group", "user_group"]:
        args.partition_mode = "user_grouped"

    # sync field cũ để tương thích toàn bộ project
    args.shard_mode = "user"

    if args.partition_mode == "user_grouped":
        args.slice_mode = "user_grouped"
        args.receraser_split_mode = "user-group"
        args.partition_method = "user_grouped"
    else:
        args.slice_mode = "random"
        args.receraser_split_mode = "random"
        args.partition_method = "random"

    if args.k is None:
        args.k = 0

    if args.slice_num is None:
        args.slice_num = args.shard_num

    # field cũ giữ lại để code khác không lỗi
    args.new_shard_mode = "user_based"
    args.new_partition_way = args.partition_mode
    args.new_slice_mode = args.slice_mode if args.method_type == "sisa" else None

    return args


# =========================================================
# VALIDATE
# =========================================================
def validate_args(args):
    if args.shard_num <= 0:
        raise ValueError("shard_num must be > 0")

    if args.slice_num <= 0:
        raise ValueError("slice_num must be > 0")

    if args.method_type == "sisa":
        if args.k < 0 or args.k >= args.slice_num:
            raise ValueError(f"k must be in [0, {args.slice_num - 1}]")

    return args


# =========================================================
# PRINT
# =========================================================
def print_header(args):
    print("=" * 100)
    print("UNLEARNING BENCHMARK")
    print("=" * 100)
    print(f"dataset_name         : {args.dataset_name}")
    print(f"model_type           : {args.model_type}")
    print(f"method_type          : {args.method_type}")
    print(f"partition_mode       : {getattr(args, 'partition_mode', None)}")
    print(f"slice_mode           : {getattr(args, 'slice_mode', None)}")
    print(f"receraser_split_mode : {getattr(args, 'receraser_split_mode', None)}")
    print(f"partition_method     : {getattr(args, 'partition_method', None)}")
    print(f"k (slice idx)        : {args.k}")
    print(f"slice_num            : {args.slice_num}")
    print(f"unlearn_users        : {args.unlearn_users}")
    print("=" * 100)