import csv
import math
import os
import random
import time
import numpy as np

# =========================================================
# GPU / CPU AUTO SETUP
# =========================================================
def setup_device():
    requested = str(os.environ.get("DEVICE", "auto")).strip().lower()
    gpu_id = str(os.environ.get("GPU_ID", "0")).strip()

    def has_gpu():
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            return r.returncode == 0 and "GPU" in r.stdout
        except Exception:
            return False

    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "CPU"

    if requested == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return f"GPU:{gpu_id}"

    if has_gpu():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return f"GPU:{gpu_id}"

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return "CPU"

DEVICE = setup_device()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import Config
from data_loader import DataLoader
from core.registry import build_model, build_method


def build_cfg(method_type='receraser', model_type='bpr'):
    cfg = Config()
    cfg.method_type = method_type
    cfg.method = method_type
    cfg.model_type = model_type
    cfg.device = DEVICE
    cfg.use_gpu = DEVICE.startswith("GPU")
    if method_type == 'receraser':
        cfg.partition_type = getattr(cfg, 'receraser_partition_type', cfg.partition_type)
    elif method_type == 'sisa':
        cfg.partition_type = getattr(cfg, 'sisa_partition_type', cfg.partition_type)
    cfg.sync_alias_fields()
    return cfg

def dcg_at_k(relevance, k):
    relevance = relevance[:k]
    if len(relevance) == 0:
        return 0.0
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))

def ndcg_at_k(ranked_items, gt_items, k):
    if len(gt_items) == 0:
        return 0.0
    relevance = [1.0 if item in gt_items else 0.0 for item in ranked_items[:k]]
    dcg = dcg_at_k(relevance, k)
    ideal_len = min(len(gt_items), k)
    ideal_relevance = [1.0] * ideal_len
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def recall_at_k(ranked_items, gt_items, k):
    if len(gt_items) == 0:
        return 0.0
    hit_count = sum(1 for item in ranked_items[:k] if item in gt_items)
    return hit_count / float(len(gt_items))

def evaluate_model(model, train_user_dict, test_user_dict, n_items, topk_list):
    valid_users = [u for u in test_user_dict if len(test_user_dict[u]) > 0]
    metrics = {}
    for k in topk_list:
        metrics[f'recall@{k}'] = 0.0
        metrics[f'ndcg@{k}'] = 0.0
    if len(valid_users) == 0:
        return metrics

    recall_sum = {k: 0.0 for k in topk_list}
    ndcg_sum = {k: 0.0 for k in topk_list}

    for u in valid_users:
        scores = np.asarray(model.predict(u), dtype=np.float64)
        if scores.shape[0] != n_items:
            raise ValueError(f'predict({u}) returned shape {scores.shape}, expected ({n_items},)')
        seen_items = set(train_user_dict.get(u, []))
        if seen_items:
            scores[list(seen_items)] = -1e18
        max_k = max(topk_list)
        top_items = np.argpartition(-scores, max_k - 1)[:max_k]
        top_items = top_items[np.argsort(-scores[top_items])].tolist()
        gt_items = set(test_user_dict[u])
        for k in topk_list:
            recall_sum[k] += recall_at_k(top_items, gt_items, k)
            ndcg_sum[k] += ndcg_at_k(top_items, gt_items, k)

    n_eval = len(valid_users)
    for k in topk_list:
        metrics[f'recall@{k}'] = recall_sum[k] / n_eval
        metrics[f'ndcg@{k}'] = ndcg_sum[k] / n_eval
    return metrics

def print_metrics(title, metrics, topk_list):
    print(f'\n[{title}]')
    for k in topk_list:
        print(f"  Recall@{k} = {metrics[f'recall@{k}']:.6f} | NDCG@{k} = {metrics[f'ndcg@{k}']:.6f}")

def pick_random_users(loader, num_runs, user_count=1, seed=2024):
    rng = random.Random(seed)
    candidates = [u for u, items in loader.train_user_dict.items() if len(items) > 0]
    if len(candidates) == 0:
        raise ValueError('No valid users to unlearn.')
    return [rng.sample(candidates, user_count) for _ in range(num_runs)]

def pick_random_interactions(loader, num_runs, interaction_count=1, seed=2024):
    rng = random.Random(seed)
    candidates = [(u, i) for u, items in loader.train_user_dict.items() for i in items]
    if len(candidates) == 0:
        raise ValueError('No valid interactions to unlearn.')
    return [rng.sample(candidates, interaction_count) for _ in range(num_runs)]

def pick_random_items(loader, num_runs, item_count=1, seed=2024):
    rng = random.Random(seed)
    item_set = set()
    for _, items in loader.train_user_dict.items():
        item_set.update(items)
    candidates = sorted(list(item_set))
    if len(candidates) == 0:
        raise ValueError('No valid items to unlearn.')
    return [rng.sample(candidates, item_count) for _ in range(num_runs)]

def get_final_model_from_method(method):
    if hasattr(method, 'get_final_model'):
        return method.get_final_model()
    if hasattr(method, 'final_model'):
        return method.final_model
    if hasattr(method, 'base_model'):
        return method.base_model
    raise ValueError('Cannot find final model from method.')

def default_stats():
    return {
        'affected_shards': [],
        'affected_from_users': [],
        'affected_from_items': [],
        'affected_from_interactions': [],
        'affected_slices': {},
        'affected_slice_start': {},
        'retrain_shard_stats': {},
        'n_affected_shards': 0,
        'total_retrain_users': 0,
        'total_retrain_items': 0,
        'total_retrain_interactions': 0,
        'retrain_ratio': 0.0,
        'affected_shard_time': 0.0,
        'agg_train_time': 0.0,
        'retrain_time': 0.0,
        'total_time': 0.0,
    }

def merge_stats(stats):
    out = default_stats()
    if isinstance(stats, dict):
        out.update(stats)
    return out

def get_display_initial_time(stats, fallback_elapsed):
    if isinstance(stats, dict):
        return float(stats.get('train_time', fallback_elapsed))
    return float(fallback_elapsed)

def get_display_retrain_time(stats, fallback_elapsed):
    if isinstance(stats, dict):
        return float(stats.get('retrain_time', fallback_elapsed))
    return float(fallback_elapsed)

def format_target(unlearn_type, target):
    if unlearn_type == 'user':
        return f'users={list(target)}'
    if unlearn_type == 'interaction':
        return f'interactions={list(target)}'
    if unlearn_type == 'item':
        return f'items={list(target)}'
    return str(target)

def print_target_banner(run_id, num_runs, unlearn_type, target):
    print('\n' + '=' * 72)
    print(f'RUN {run_id}/{num_runs}')
    print(f'UNLEARN TYPE : {unlearn_type}')
    print(f'TARGET       : {format_target(unlearn_type, target)}')
    print('=' * 72)

def print_method_breakdown(method_name, stats):
    stats = merge_stats(stats)
    print(f'\n[{method_name} BREAKDOWN]')
    print(f"  affected_shards              = {stats.get('affected_shards', [])}")
    print(f"  affected_from_users          = {stats.get('affected_from_users', [])}")
    print(f"  affected_from_items          = {stats.get('affected_from_items', [])}")
    print(f"  affected_from_interactions   = {stats.get('affected_from_interactions', [])}")
    print(f"  affected_slices              = {stats.get('affected_slices', {})}")
    print(f"  affected_slice_start         = {stats.get('affected_slice_start', {})}")
    print(f"  n_affected_shards            = {stats.get('n_affected_shards', 0)}")
    print(f"  total_retrain_users          = {stats.get('total_retrain_users', 0)}")
    print(f"  total_retrain_items          = {stats.get('total_retrain_items', 0)}")
    print(f"  total_retrain_interactions   = {stats.get('total_retrain_interactions', 0)}")
    print(f"  retrain_ratio                = {stats.get('retrain_ratio', 0.0)}")
    print(f"  affected_shard_time          = {stats.get('affected_shard_time', 0.0):.4f}s")
    print(f"  agg_train_time               = {stats.get('agg_train_time', 0.0):.4f}s")
    print(f"  retrain_time                 = {stats.get('retrain_time', 0.0):.4f}s")
    print(f"  total_time                   = {stats.get('total_time', 0.0):.4f}s")
    retrain_shard_stats = stats.get('retrain_shard_stats', {})
    if retrain_shard_stats:
        print('  retrain_shard_stats:')
        for sid in sorted(retrain_shard_stats.keys()):
            shard_stat = retrain_shard_stats[sid]
            print(f"    shard={sid} | users={shard_stat.get('users', 0)} | items={shard_stat.get('items', 0)} | interactions={shard_stat.get('interactions', 0)}")

def build_loader_model_method(cfg):
    loader = DataLoader(cfg)
    ModelClass = build_model(cfg, loader.n_users, loader.n_items)
    method = build_method(cfg, loader, ModelClass)
    return loader, method

def run_retrain_initial(cfg):
    print('\n================ RETRAIN: INITIAL TRAIN ================')
    loader = DataLoader(cfg)
    ModelClass = build_model(cfg, loader.n_users, loader.n_items)
    model = ModelClass(cfg, loader.n_users, loader.n_items)
    t0 = time.time()
    fit_stats = model.fit(loader.train_user_dict, epochs=cfg.epochs)
    elapsed = time.time() - t0
    metrics = evaluate_model(model, loader.train_user_dict, loader.test_user_dict, loader.n_items, cfg.topk_list)
    stats = {'train_time': elapsed, 'retrain_time': elapsed, 'total_time': elapsed}
    display_time = stats.get('train_time', elapsed)
    print(f'[RETRAIN][INITIAL] train_time={display_time:.4f}s')
    print(f'[RETRAIN][INITIAL] raw_elapsed={elapsed:.4f}s')
    print(f'[RETRAIN][INITIAL] last_stats={fit_stats}')
    print_metrics('RETRAIN INITIAL ACCURACY', metrics, cfg.topk_list)
    return loader, model, display_time, metrics, stats

def run_method_initial(cfg, method_name):
    print(f'\n================ {method_name.upper()}: INITIAL TRAIN ================')
    loader, method = build_loader_model_method(cfg)
    t0 = time.time()
    stats = method.initial_train()
    elapsed = time.time() - t0
    display_time = get_display_initial_time(stats, elapsed)
    final_model = get_final_model_from_method(method)
    metrics = evaluate_model(final_model, loader.train_user_dict, loader.test_user_dict, loader.n_items, cfg.topk_list)
    print(f'[{method_name.upper()}][INITIAL] train_time={display_time:.4f}s')
    print(f'[{method_name.upper()}][INITIAL] raw_elapsed={elapsed:.4f}s')
    print(f'[{method_name.upper()}][INITIAL] stats={stats}')
    print_metrics(f'{method_name.upper()} INITIAL ACCURACY', metrics, cfg.topk_list)
    if isinstance(stats, dict):
        print(f'\n[{method_name.upper()}][INITIAL][BREAKDOWN]')
        print(f"  all_shard_train_time = {stats.get('all_shard_train_time', -1):.4f}s")
        print(f"  agg_train_time       = {stats.get('agg_train_time', -1):.4f}s")
        print(f"  train_time           = {stats.get('train_time', -1):.4f}s")
        print(f"  total_time           = {stats.get('total_time', -1):.4f}s")
    return loader, method, display_time, metrics, stats

def run_retrain_unlearn_user(cfg, target_users):
    loader = DataLoader(cfg)
    target_user_set = set(target_users)
    print(f'[RETRAIN] Unlearn target users: {list(target_users)}')
    train_after = {u: items for u, items in loader.train_user_dict.items() if u not in target_user_set and len(items) > 0}
    ModelClass = build_model(cfg, loader.n_users, loader.n_items)
    model = ModelClass(cfg, loader.n_users, loader.n_items)
    t0 = time.time()
    _ = model.fit(train_after, epochs=cfg.epochs)
    elapsed = time.time() - t0
    metrics = evaluate_model(model, train_after, loader.test_user_dict, loader.n_items, cfg.topk_list)
    stats = {'train_time': elapsed, 'retrain_time': elapsed, 'total_time': elapsed, 'affected_shards': ['full_retrain'], 'n_affected_shards': 1, 'total_retrain_users': len(train_after), 'total_retrain_items': len({i for items in train_after.values() for i in items}), 'total_retrain_interactions': sum(len(v) for v in train_after.values())}
    display_time = stats.get('retrain_time', elapsed)
    return display_time, metrics, stats

def run_retrain_unlearn_interaction(cfg, target_interactions):
    loader = DataLoader(cfg)
    interaction_set = set((int(u), int(i)) for u, i in target_interactions)
    print(f'[RETRAIN] Unlearn target interactions: {list(target_interactions)}')
    train_after = {}
    for u, items in loader.train_user_dict.items():
        new_items = [i for i in items if (u, i) not in interaction_set]
        if len(new_items) > 0:
            train_after[u] = new_items
    ModelClass = build_model(cfg, loader.n_users, loader.n_items)
    model = ModelClass(cfg, loader.n_users, loader.n_items)
    t0 = time.time()
    _ = model.fit(train_after, epochs=cfg.epochs)
    elapsed = time.time() - t0
    metrics = evaluate_model(model, train_after, loader.test_user_dict, loader.n_items, cfg.topk_list)
    stats = {'train_time': elapsed, 'retrain_time': elapsed, 'total_time': elapsed, 'affected_shards': ['full_retrain'], 'n_affected_shards': 1, 'total_retrain_users': len(train_after), 'total_retrain_items': len({i for items in train_after.values() for i in items}), 'total_retrain_interactions': sum(len(v) for v in train_after.values())}
    display_time = stats.get('retrain_time', elapsed)
    return display_time, metrics, stats

def run_retrain_unlearn_item(cfg, target_items):
    loader = DataLoader(cfg)
    item_set = set(target_items)
    print(f'[RETRAIN] Unlearn target items: {list(target_items)}')
    train_after = {}
    for u, items in loader.train_user_dict.items():
        new_items = [i for i in items if i not in item_set]
        if len(new_items) > 0:
            train_after[u] = new_items
    ModelClass = build_model(cfg, loader.n_users, loader.n_items)
    model = ModelClass(cfg, loader.n_users, loader.n_items)
    t0 = time.time()
    _ = model.fit(train_after, epochs=cfg.epochs)
    elapsed = time.time() - t0
    metrics = evaluate_model(model, train_after, loader.test_user_dict, loader.n_items, cfg.topk_list)
    stats = {'train_time': elapsed, 'retrain_time': elapsed, 'total_time': elapsed, 'affected_shards': ['full_retrain'], 'n_affected_shards': 1, 'total_retrain_users': len(train_after), 'total_retrain_items': len({i for items in train_after.values() for i in items}), 'total_retrain_interactions': sum(len(v) for v in train_after.values())}
    display_time = stats.get('retrain_time', elapsed)
    return display_time, metrics, stats

def _reset_loader_for_unlearn(loader):
    if hasattr(loader, 'reset_all_train_state'):
        loader.reset_all_train_state()
    elif hasattr(loader, 'reset_partition_state'):
        loader.reset_partition_state()
    elif hasattr(loader, 'reset_global_train_data'):
        loader.reset_global_train_data()

def run_method_unlearn_user(loader, method, cfg, target_users, method_name):
    print(f'\n================ {method_name.upper()}: UNLEARN USER ================')
    print(f'[{method_name.upper()}] target users = {list(target_users)}')
    _reset_loader_for_unlearn(loader)
    target_user_set = set(target_users)
    t0 = time.time()
    final_model, stats = method.unlearn(users_to_remove=target_users)
    elapsed = time.time() - t0
    display_time = get_display_retrain_time(stats, elapsed)
    train_after = {u: items for u, items in loader.train_user_dict.items() if u not in target_user_set and len(items) > 0}
    metrics = evaluate_model(final_model, train_after, loader.test_user_dict, loader.n_items, cfg.topk_list)
    print(f'[{method_name.upper()}][UNLEARN USERS={target_users}] retrain_time={display_time:.4f}s')
    print(f'[{method_name.upper()}][UNLEARN USERS={target_users}] raw_elapsed={elapsed:.4f}s')
    print(f'[{method_name.upper()}][UNLEARN USER] stats={stats}')
    print_metrics(f'{method_name.upper()} AFTER USER UNLEARN', metrics, cfg.topk_list)
    print_method_breakdown(method_name.upper(), stats)
    return display_time, merge_stats(stats), metrics

def run_method_unlearn_interaction(loader, method, cfg, target_interactions, method_name):
    print(f'\n================ {method_name.upper()}: UNLEARN INTERACTION ================')
    print(f'[{method_name.upper()}] target interactions = {list(target_interactions)}')
    _reset_loader_for_unlearn(loader)
    interaction_set = set((int(u), int(i)) for u, i in target_interactions)
    t0 = time.time()
    final_model, stats = method.unlearn(interactions_to_remove=target_interactions)
    elapsed = time.time() - t0
    display_time = get_display_retrain_time(stats, elapsed)
    train_after = {}
    for u, items in loader.train_user_dict.items():
        new_items = [i for i in items if (u, i) not in interaction_set]
        if len(new_items) > 0:
            train_after[u] = new_items
    metrics = evaluate_model(final_model, train_after, loader.test_user_dict, loader.n_items, cfg.topk_list)
    print(f'[{method_name.upper()}][UNLEARN INTERACTIONS={target_interactions}] retrain_time={display_time:.4f}s')
    print(f'[{method_name.upper()}][UNLEARN INTERACTIONS={target_interactions}] raw_elapsed={elapsed:.4f}s')
    print(f'[{method_name.upper()}][UNLEARN INTERACTION] stats={stats}')
    print_metrics(f'{method_name.upper()} AFTER INTERACTION UNLEARN', metrics, cfg.topk_list)
    print_method_breakdown(method_name.upper(), stats)
    return display_time, merge_stats(stats), metrics

def run_method_unlearn_item(loader, method, cfg, target_items, method_name):
    print(f'\n================ {method_name.upper()}: UNLEARN ITEM ================')
    print(f'[{method_name.upper()}] target items = {list(target_items)}')
    _reset_loader_for_unlearn(loader)
    item_set = set(target_items)
    t0 = time.time()
    final_model, stats = method.unlearn(items_to_remove=target_items)
    elapsed = time.time() - t0
    display_time = get_display_retrain_time(stats, elapsed)
    train_after = {}
    for u, items in loader.train_user_dict.items():
        new_items = [i for i in items if i not in item_set]
        if len(new_items) > 0:
            train_after[u] = new_items
    metrics = evaluate_model(final_model, train_after, loader.test_user_dict, loader.n_items, cfg.topk_list)
    print(f'[{method_name.upper()}][UNLEARN ITEMS={target_items}] retrain_time={display_time:.4f}s')
    print(f'[{method_name.upper()}][UNLEARN ITEMS={target_items}] raw_elapsed={elapsed:.4f}s')
    print(f'[{method_name.upper()}][UNLEARN ITEM] stats={stats}')
    print_metrics(f'{method_name.upper()} AFTER ITEM UNLEARN', metrics, cfg.topk_list)
    print_method_breakdown(method_name.upper(), stats)
    return display_time, merge_stats(stats), metrics

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_csv_path(cfg):
    csv_dir = getattr(cfg, 'result_dir', 'results')
    ensure_dir(csv_dir)
    unlearn_type = getattr(cfg, 'unlearn_type', 'user')
    model_type = getattr(cfg, 'model_type', 'bpr')
    partition_type = getattr(cfg, 'partition_type', 'user_based')
    dataset_name = getattr(cfg, 'dataset_name', 'dataset')
    num_runs = getattr(cfg, 'unlearn_eval_runs', 5)
    shard_num = getattr(cfg, 'shard_num', 10)
    filename = f"{dataset_name}__{model_type}__{partition_type}__{unlearn_type}__runs{num_runs}__shard{shard_num}.csv"
    return os.path.join(csv_dir, filename)

def build_csv_header(topk_list):
    header = ['run_id', 'unlearn_type', 'target', 'method', 'time', 'n_affected_shards', 'total_retrain_users', 'total_retrain_items', 'total_retrain_interactions', 'retrain_ratio', 'affected_shard_time', 'agg_train_time', 'retrain_time', 'total_time']
    for k in topk_list:
        header.append(f'recall@{k}')
    for k in topk_list:
        header.append(f'ndcg@{k}')
    return header

def metrics_to_row(run_id, unlearn_type, target, method_name, time_value, stats, metrics, topk_list):
    stats = merge_stats(stats)
    row = {'run_id': run_id, 'unlearn_type': unlearn_type, 'target': str(target), 'method': method_name, 'time': time_value, 'n_affected_shards': stats.get('n_affected_shards', 0), 'total_retrain_users': stats.get('total_retrain_users', 0), 'total_retrain_items': stats.get('total_retrain_items', 0), 'total_retrain_interactions': stats.get('total_retrain_interactions', 0), 'retrain_ratio': stats.get('retrain_ratio', 0.0), 'affected_shard_time': stats.get('affected_shard_time', 0.0), 'agg_train_time': stats.get('agg_train_time', 0.0), 'retrain_time': stats.get('retrain_time', time_value), 'total_time': stats.get('total_time', time_value)}
    for k in topk_list:
        row[f'recall@{k}'] = metrics.get(f'recall@{k}', 0.0)
    for k in topk_list:
        row[f'ndcg@{k}'] = metrics.get(f'ndcg@{k}', 0.0)
    return row

def write_csv(csv_path, rows, topk_list):
    header = build_csv_header(topk_list)
    numeric_fields = ["time", "n_affected_shards", "total_retrain_users", "total_retrain_items", "total_retrain_interactions", "retrain_ratio", "affected_shard_time", "agg_train_time", "retrain_time", "total_time"]
    for k in topk_list:
        numeric_fields.append(f"recall@{k}")
        numeric_fields.append(f"ndcg@{k}")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    avg_rows = []
    for method_name, method_rows in grouped.items():
        avg = {k: "" for k in header}
        avg["run_id"] = "AVG"
        avg["method"] = method_name
        for field in numeric_fields:
            values = [float(r[field]) for r in method_rows if r[field] != ""]
            if len(values) > 0:
                avg[field] = sum(values) / len(values)
        avg_rows.append(avg)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow({})
        for avg in avg_rows:
            writer.writerow(avg)

def print_average_summary(rows, topk_list):
    if len(rows) == 0:
        return
    grouped = {}
    for row in rows:
        grouped.setdefault(row['method'], []).append(row)
    print('\n================ AVERAGE OVER RUNS ================')
    for method_name, method_rows in grouped.items():
        n = len(method_rows)
        avg_time = sum(float(r['time']) for r in method_rows) / n
        avg_aff = sum(float(r['n_affected_shards']) for r in method_rows) / n
        avg_ratio = sum(float(r['retrain_ratio']) for r in method_rows) / n
        avg_shard_time = sum(float(r['affected_shard_time']) for r in method_rows) / n
        avg_agg_time = sum(float(r['agg_train_time']) for r in method_rows) / n
        print(f'\n[{method_name}]')
        print(f'  avg_time               = {avg_time:.4f}s')
        print(f'  avg_affected_shards    = {avg_aff:.4f}')
        print(f'  avg_retrain_ratio      = {avg_ratio:.6f}')
        print(f'  avg_affected_shard_time= {avg_shard_time:.4f}s')
        print(f'  avg_agg_train_time     = {avg_agg_time:.4f}s')
        for k in topk_list:
            avg_recall = sum(float(r[f'recall@{k}']) for r in method_rows) / n
            avg_ndcg = sum(float(r[f'ndcg@{k}']) for r in method_rows) / n
            print(f'  Recall@{k}={avg_recall:.6f} | NDCG@{k}={avg_ndcg:.6f}')

def main():
    print(f'[DEVICE] Using {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})')
    cfg_base = Config()
    cfg_base.device = DEVICE
    cfg_base.use_gpu = DEVICE.startswith("GPU")
    cfg_base.sync_alias_fields()
    method_mode = str(getattr(cfg_base, 'method_type', 'all')).lower()
    model_type = str(getattr(cfg_base, 'model_type', 'bpr')).lower()
    cfg_retrain = build_cfg(method_type='retrain', model_type=model_type)
    cfg_sisa = build_cfg(method_type='sisa', model_type=model_type)
    cfg_rec = build_cfg(method_type='receraser', model_type=model_type)
    random.seed(getattr(cfg_base, 'seed', 2024))

    retrain_loader = retrain_model = None
    sisa_loader = sisa_method = None
    rec_loader = rec_method = None
    retrain_initial_time = retrain_initial_metrics = None
    sisa_initial_time = sisa_initial_metrics = None
    rec_initial_time = rec_initial_metrics = None

    if method_mode in ['retrain', 'all']:
        retrain_loader, retrain_model, retrain_initial_time, retrain_initial_metrics, _ = run_retrain_initial(cfg_retrain)
    if method_mode in ['sisa', 'all']:
        sisa_loader, sisa_method, sisa_initial_time, sisa_initial_metrics, _ = run_method_initial(cfg_sisa, 'sisa')
    if method_mode in ['receraser', 'all']:
        rec_loader, rec_method, rec_initial_time, rec_initial_metrics, _ = run_method_initial(cfg_rec, 'receraser')

    print('\n================ INITIAL TRAIN COMPARISON ================')
    if retrain_initial_time is not None:
        print(f'Retrain initial time   : {retrain_initial_time:.4f}s')
        print_metrics('RETRAIN INITIAL', retrain_initial_metrics, cfg_base.topk_list)
    if sisa_initial_time is not None:
        print(f'SISA initial time      : {sisa_initial_time:.4f}s')
        print_metrics('SISA INITIAL', sisa_initial_metrics, cfg_base.topk_list)
    if rec_initial_time is not None:
        print(f'RecEraser initial time : {rec_initial_time:.4f}s')
        print_metrics('RECERASER INITIAL', rec_initial_metrics, cfg_base.topk_list)

    num_runs = int(getattr(cfg_base, 'unlearn_eval_runs', 5))
    unlearn_type = str(getattr(cfg_base, 'unlearn_type', 'user')).lower()
    seed = int(getattr(cfg_base, 'unlearn_seed', 2024))
    user_count = int(getattr(cfg_base, 'unlearn_user_count', 1))
    interaction_count = int(getattr(cfg_base, 'unlearn_interaction_count', 1))
    item_count = int(getattr(cfg_base, 'unlearn_item_count', 1))

    target_loader = rec_loader if rec_loader is not None else sisa_loader if sisa_loader is not None else retrain_loader
    if target_loader is None:
        raise ValueError('No loader available to build unlearning targets.')

    if unlearn_type == 'user':
        targets = pick_random_users(target_loader, num_runs, user_count=user_count, seed=seed)
    elif unlearn_type == 'interaction':
        targets = pick_random_interactions(target_loader, num_runs, interaction_count=interaction_count, seed=seed)
    elif unlearn_type == 'item':
        targets = pick_random_items(target_loader, num_runs, item_count=item_count, seed=seed)
    else:
        raise ValueError("unlearn_type must be 'user', 'interaction', or 'item'")

    csv_rows = []
    for run_id, target in enumerate(targets, start=1):
        print_target_banner(run_id, num_runs, unlearn_type, target)
        retrain_time = retrain_metrics = retrain_stats = None
        sisa_time = sisa_metrics = sisa_stats = None
        rec_time = rec_metrics = rec_stats = None

        if unlearn_type == 'user':
            if method_mode in ['retrain', 'all']:
                retrain_time, retrain_metrics, retrain_stats = run_retrain_unlearn_user(cfg_retrain, target)
            if method_mode in ['sisa', 'all']:
                sisa_time, sisa_stats, sisa_metrics = run_method_unlearn_user(sisa_loader, sisa_method, cfg_sisa, target, 'sisa')
            if method_mode in ['receraser', 'all']:
                rec_time, rec_stats, rec_metrics = run_method_unlearn_user(rec_loader, rec_method, cfg_rec, target, 'receraser')
        elif unlearn_type == 'interaction':
            if method_mode in ['retrain', 'all']:
                retrain_time, retrain_metrics, retrain_stats = run_retrain_unlearn_interaction(cfg_retrain, target)
            if method_mode in ['sisa', 'all']:
                sisa_time, sisa_stats, sisa_metrics = run_method_unlearn_interaction(sisa_loader, sisa_method, cfg_sisa, target, 'sisa')
            if method_mode in ['receraser', 'all']:
                rec_time, rec_stats, rec_metrics = run_method_unlearn_interaction(rec_loader, rec_method, cfg_rec, target, 'receraser')
        elif unlearn_type == 'item':
            if method_mode in ['retrain', 'all']:
                retrain_time, retrain_metrics, retrain_stats = run_retrain_unlearn_item(cfg_retrain, target)
            if method_mode in ['sisa', 'all']:
                sisa_time, sisa_stats, sisa_metrics = run_method_unlearn_item(sisa_loader, sisa_method, cfg_sisa, target, 'sisa')
            if method_mode in ['receraser', 'all']:
                rec_time, rec_stats, rec_metrics = run_method_unlearn_item(rec_loader, rec_method, cfg_rec, target, 'receraser')

        print(f'\n---------------- RUN {run_id} RESULT SUMMARY ----------------')
        if retrain_time is not None:
            print(f'Retrain time   : {retrain_time:.4f}s')
            print_metrics('RETRAIN', retrain_metrics, cfg_base.topk_list)
            print_method_breakdown('RETRAIN', retrain_stats)
        if sisa_time is not None:
            print(f'SISA time      : {sisa_time:.4f}s')
            print_metrics('SISA', sisa_metrics, cfg_base.topk_list)
            print_method_breakdown('SISA', sisa_stats)
        if rec_time is not None:
            print(f'RecEraser time : {rec_time:.4f}s')
            print_metrics('RECERASER', rec_metrics, cfg_base.topk_list)
            print_method_breakdown('RECERASER', rec_stats)

        if method_mode in ['retrain', 'all']:
            csv_rows.append(metrics_to_row(run_id, unlearn_type, target, 'Retrain', retrain_time, retrain_stats, retrain_metrics, cfg_base.topk_list))
        if method_mode in ['sisa', 'all']:
            csv_rows.append(metrics_to_row(run_id, unlearn_type, target, 'SISA', sisa_time, sisa_stats, sisa_metrics, cfg_base.topk_list))
        if method_mode in ['receraser', 'all']:
            csv_rows.append(metrics_to_row(run_id, unlearn_type, target, 'RecEraser', rec_time, rec_stats, rec_metrics, cfg_base.topk_list))

    csv_path = build_csv_path(cfg_base)
    write_csv(csv_path, csv_rows, cfg_base.topk_list)
    print(f'\n[CSV SAVED] {csv_path}')
    print_average_summary(csv_rows, cfg_base.topk_list)

if __name__ == '__main__':
    main()
