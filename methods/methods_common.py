import os
import pickle as pkl
import numpy as np


# =========================================================
# BASIC HELPERS
# =========================================================
def count_interactions(user_dict):
    return sum(len(items) for items in user_dict.values())


def merge_slices(slices, end_stage):
    merged = {}
    for s in range(end_stage + 1):
        for u, items in slices[s].items():
            merged.setdefault(u, []).extend(items)
    return merged


# =========================================================
# CHECKPOINT SAVE / LOAD
# =========================================================
def save_state(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(state, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_state(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pkl.load(f)


# =========================================================
# STATE AGGREGATION
# =========================================================
def average_states(states):
    """
    Robust average for model states:
    - supports numpy arrays
    - supports python scalars
    - supports list of arrays
    """
    if not states:
        return None

    keys = list(states[0].keys())
    out = {}

    for k in keys:
        vals = [s[k] for s in states]

        # case 1: list of arrays/tensors
        if isinstance(vals[0], list):
            merged = []
            for idx in range(len(vals[0])):
                elems = [np.asarray(v[idx], dtype=np.float64) for v in vals]
                merged.append(np.mean(np.stack(elems, axis=0), axis=0))
            out[k] = merged
            continue

        # case 2: scalar
        if np.isscalar(vals[0]):
            out[k] = float(np.mean(vals))
            continue

        # case 3: ndarray / tensor-like
        arrs = [np.asarray(v, dtype=np.float64) for v in vals]
        out[k] = np.mean(np.stack(arrs, axis=0), axis=0)

    return out


# =========================================================
# EMBEDDING PRETRAIN UTILS
# =========================================================
def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_real_model(model):
    real_model = model
    for attr in ["model", "_model", "net", "backbone"]:
        if hasattr(real_model, attr):
            real_model = getattr(real_model, attr)
            break
    return real_model


def _reduce_embedding_if_needed(emb):
    emb = np.asarray(emb)

    if emb.ndim == 2:
        return emb

    if emb.ndim == 3:
        # RecEraser local embeddings: [n_entity, n_local, emb_dim]
        return np.mean(emb, axis=1)

    raise TypeError(f"Unsupported embedding ndim={emb.ndim}, shape={emb.shape}")


def extract_user_item_embeddings(model):
    if hasattr(model, "get_state"):
        try:
            state = model.get_state()
            if isinstance(state, dict):
                if "user_emb" in state and "item_emb" in state:
                    user_emb = _reduce_embedding_if_needed(np.asarray(state["user_emb"]))
                    item_emb = _reduce_embedding_if_needed(np.asarray(state["item_emb"]))
                    return user_emb, item_emb

                user_keys = [
                    "user_embedding:0",
                    "user_emb:0",
                    "user_embedding",
                    "user_emb",
                ]
                item_keys = [
                    "item_embedding:0",
                    "item_emb:0",
                    "item_embedding",
                    "item_emb",
                ]

                user_obj = None
                item_obj = None

                for kk in user_keys:
                    if kk in state:
                        user_obj = state[kk]
                        break

                for kk in item_keys:
                    if kk in state:
                        item_obj = state[kk]
                        break

                if user_obj is not None and item_obj is not None:
                    user_emb = _reduce_embedding_if_needed(np.asarray(user_obj))
                    item_emb = _reduce_embedding_if_needed(np.asarray(item_obj))
                    return user_emb, item_emb
        except Exception:
            pass

    real_model = _get_real_model(model)

    user_candidates = ["user_emb", "user_embedding", "user_embeddings", "embedding_user"]
    item_candidates = ["item_emb", "item_embedding", "item_embeddings", "embedding_item"]

    user_obj = None
    item_obj = None

    for name in user_candidates:
        if hasattr(real_model, name):
            user_obj = getattr(real_model, name)
            break

    for name in item_candidates:
        if hasattr(real_model, name):
            item_obj = getattr(real_model, name)
            break

    if user_obj is None or item_obj is None:
        raise AttributeError("Không tìm thấy user/item embedding trong model.")

    user_weight = user_obj.weight if hasattr(user_obj, "weight") else user_obj
    item_weight = item_obj.weight if hasattr(item_obj, "weight") else item_obj

    user_emb = _reduce_embedding_if_needed(_to_numpy(user_weight))
    item_emb = _reduce_embedding_if_needed(_to_numpy(item_weight))

    return user_emb, item_emb


def save_pretrain_embeddings(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    user_emb, item_emb = extract_user_item_embeddings(model)

    user_emb = np.asarray(user_emb)
    item_emb = np.asarray(item_emb)

    if user_emb.ndim != 2 or item_emb.ndim != 2:
        raise TypeError(
            f"user_emb/item_emb must be 2D arrays after reduction, got shapes: "
            f"{getattr(user_emb, 'shape', None)} and {getattr(item_emb, 'shape', None)}"
        )

    user_dict = {int(i): user_emb[i] for i in range(user_emb.shape[0])}
    item_dict = {int(i): item_emb[i] for i in range(item_emb.shape[0])}

    user_path = os.path.join(save_dir, "user_pretrain.pk")
    item_path = os.path.join(save_dir, "item_pretrain.pk")

    with open(user_path, "wb") as f:
        pkl.dump(user_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(item_path, "wb") as f:
        pkl.dump(item_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("[SAVE] user_pretrain ->", user_path)
    print("[SAVE] item_pretrain ->", item_path)


def load_pretrain_embeddings(load_dir):
    user_path = os.path.join(load_dir, "user_pretrain.pk")
    item_path = os.path.join(load_dir, "item_pretrain.pk")

    if not os.path.exists(user_path) or not os.path.exists(item_path):
        return None, None

    with open(user_path, "rb") as f:
        user_dict = pickle.load(f)

    with open(item_path, "rb") as f:
        item_dict = pickle.load(f)

    print("[LOAD] user_pretrain <-", user_path)
    print("[LOAD] item_pretrain <-", item_path)
    return user_dict, item_dict


def has_pretrain_embeddings(load_dir):
    user_path = os.path.join(load_dir, "user_pretrain.pk")
    item_path = os.path.join(load_dir, "item_pretrain.pk")
    return os.path.exists(user_path) and os.path.exists(item_path)


def pretrain_files_exist(save_dir, require_item=True):
    user_path = os.path.join(save_dir, "user_pretrain.pk")
    item_path = os.path.join(save_dir, "item_pretrain.pk")

    if require_item:
        return os.path.exists(user_path) and os.path.exists(item_path)
    return os.path.exists(user_path)