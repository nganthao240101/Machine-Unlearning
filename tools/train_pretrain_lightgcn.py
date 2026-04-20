import os
import time
import pickle as pkl
import random
import argparse

import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGCN pretrain embeddings for RecEraser")

    parser.add_argument("--dataset", type=str, default="ml-1m",
                        help="Dataset name: ml-1m | ml-10m | yelp2018")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--embed_size", type=int, default=64,
                        help="Embedding size")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--reg", type=float, default=1e-3,
                        help="L2 regularization")
    parser.add_argument("--gcn_layers", type=int, default=2,
                        help="Number of LightGCN layers")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--topk", type=str, default="10,20,50",
                        help="Top-K list, e.g. 10,20,50")

    return parser.parse_args()


# =========================================================
# DATA
# =========================================================
class LightGCNData:
    def __init__(self, train_path, test_path, seed=2024):
        self.train_path = train_path
        self.test_path = test_path
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.train_user_dict = {}
        self.test_user_dict = {}

        self.n_users = 0
        self.n_items = 0
        self.n_train = 0
        self.n_test = 0

        self.norm_adj = None

        self._load()
        self.norm_adj = self._build_norm_adj()

    def _read_user_item_file(self, path):
        d = {}
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

                d[u] = items
                max_u = max(max_u, u)
                if items:
                    max_i = max(max_i, max(items))
                    total += len(items)

        return d, max_u, max_i, total

    def _load(self):
        train_d, mu1, mi1, n_train = self._read_user_item_file(self.train_path)
        test_d, mu2, mi2, n_test = self._read_user_item_file(self.test_path)

        self.train_user_dict = train_d
        self.test_user_dict = test_d

        self.n_users = max(mu1, mu2) + 1
        self.n_items = max(mi1, mi2) + 1
        self.n_train = n_train
        self.n_test = n_test

        print("[DATA]")
        print(f"  train_path   : {self.train_path}")
        print(f"  test_path    : {self.test_path}")
        print(f"  n_users      : {self.n_users}")
        print(f"  n_items      : {self.n_items}")
        print(f"  n_train      : {self.n_train}")
        print(f"  n_test       : {self.n_test}")

    def _build_norm_adj(self):
        n_nodes = self.n_users + self.n_items
        rows = []
        cols = []
        data = []

        for u, items in self.train_user_dict.items():
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

    def sample(self, batch_size):
        users = list(self.train_user_dict.keys())

        if batch_size <= len(users):
            batch_users = random.sample(users, batch_size)
        else:
            batch_users = [random.choice(users) for _ in range(batch_size)]

        pos_items = []
        neg_items = []

        for u in batch_users:
            pos_list = self.train_user_dict[u]
            pos_i = random.choice(pos_list)
            pos_items.append(pos_i)

            while True:
                neg_i = random.randint(0, self.n_items - 1)
                if neg_i not in self.train_user_dict[u]:
                    neg_items.append(neg_i)
                    break

        return (
            np.asarray(batch_users, dtype=np.int32),
            np.asarray(pos_items, dtype=np.int32),
            np.asarray(neg_items, dtype=np.int32),
        )


# =========================================================
# MODEL
# =========================================================
class LightGCN:
    def __init__(self, n_users, n_items, norm_adj, emb_dim=64, lr=0.001, reg=1e-3, n_layers=2):
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.emb_dim = emb_dim
        self.lr = lr
        self.reg = reg
        self.n_layers = n_layers
        self.n_fold = 10

        self.users = tf.placeholder(tf.int32, shape=(None,), name="users")
        self.pos_items = tf.placeholder(tf.int32, shape=(None,), name="pos_items")
        self.neg_items = tf.placeholder(tf.int32, shape=(None,), name="neg_items")

        initializer = tf.random_normal_initializer(stddev=0.01)

        self.user_embedding = tf.Variable(
            initializer([self.n_users, self.emb_dim]),
            name="user_embedding"
        )
        self.item_embedding = tf.Variable(
            initializer([self.n_items, self.emb_dim]),
            name="item_embedding"
        )

        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.user_embedding, self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.item_embedding, self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.item_embedding, self.neg_items)

        self.batch_ratings = tf.matmul(
            self.u_g_embeddings, self.pos_i_g_embeddings,
            transpose_a=False, transpose_b=True
        )

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(
            self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings
        )
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.vstack((coo.row, coo.col)).transpose().astype(np.int64)
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_lightgcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.user_embedding, self.item_embedding], axis=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, axis=0)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = tf.stack(all_embeddings, axis=1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], axis=0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(self.u_g_embeddings_pre) +
            tf.nn.l2_loss(self.pos_i_g_embeddings_pre) +
            tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        )
        regularizer = regularizer / tf.cast(tf.maximum(tf.shape(users)[0], 1), tf.float32)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.reg * regularizer
        reg_loss = tf.constant(0.0, tf.float32)

        return mf_loss, emb_loss, reg_loss


# =========================================================
# EVAL
# =========================================================
def evaluate_model(sess, model, data, topk_list):
    recalls = [0.0 for _ in topk_list]
    precisions = [0.0 for _ in topk_list]
    ndcgs = [0.0 for _ in topk_list]

    test_users = [u for u in data.test_user_dict.keys() if len(data.test_user_dict[u]) > 0]
    n_eval = 0

    for u in test_users:
        gt_items = data.test_user_dict[u]
        if len(gt_items) == 0:
            continue

        user_arr = np.full(shape=(data.n_items,), fill_value=u, dtype=np.int32)
        item_arr = np.arange(data.n_items, dtype=np.int32)
        dummy_neg = np.zeros(data.n_items, dtype=np.int32)

        scores = sess.run(
            model.batch_ratings,
            feed_dict={
                model.users: user_arr,
                model.pos_items: item_arr,
                model.neg_items: dummy_neg,
            }
        )
        scores = np.diag(scores)

        for seen_i in data.train_user_dict.get(u, []):
            scores[seen_i] = -1e12

        ranking = np.argsort(-scores)

        for idx, k in enumerate(topk_list):
            topk_items = ranking[:k]
            hits = [1 if i in gt_items else 0 for i in topk_items]

            hit_count = sum(hits)
            recalls[idx] += hit_count / max(len(gt_items), 1)
            precisions[idx] += hit_count / k

            dcg = 0.0
            for rank, item in enumerate(topk_items):
                if item in gt_items:
                    dcg += 1.0 / np.log2(rank + 2)

            ideal_hits = min(len(gt_items), k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs[idx] += (dcg / idcg) if idcg > 0 else 0.0

        n_eval += 1

    if n_eval == 0:
        return {
            "recall": [0.0 for _ in topk_list],
            "precision": [0.0 for _ in topk_list],
            "ndcg": [0.0 for _ in topk_list],
        }

    return {
        "recall": [x / n_eval for x in recalls],
        "precision": [x / n_eval for x in precisions],
        "ndcg": [x / n_eval for x in ndcgs],
    }


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_path = os.path.join(dataset_dir, "train.txt")
    test_path = os.path.join(dataset_dir, "test.txt")

    topk_list = [int(x) for x in args.topk.split(",")]

    data = LightGCNData(train_path, test_path, seed=args.seed)

    print("use the pre adjacency matrix")

    model = LightGCN(
        n_users=data.n_users,
        n_items=data.n_items,
        norm_adj=data.norm_adj,
        emb_dim=args.embed_size,
        lr=args.lr,
        reg=args.reg,
        n_layers=args.gcn_layers,
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    t0 = time.time()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("\n[TRAIN LIGHTGCN PRETRAIN]")
        print(f"  dataset     : {args.dataset}")
        print(f"  epochs      : {args.epochs}")
        print(f"  emb_dim     : {args.embed_size}")
        print(f"  batch_size  : {args.batch_size}")
        print(f"  lr          : {args.lr}")
        print(f"  reg         : {args.reg}")
        print(f"  gcn_layers  : {args.gcn_layers}")
        print(f"  topk        : {topk_list}")

        for epoch in range(args.epochs):
            t1 = time.time()
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0

            n_batch = data.n_train // args.batch_size + 1

            for _ in range(n_batch):
                users, pos_items, neg_items = data.sample(args.batch_size)
                _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                    [model.opt, model.loss, model.mf_loss, model.emb_loss],
                    feed_dict={
                        model.users: users,
                        model.pos_items: pos_items,
                        model.neg_items: neg_items,
                    }
                )
                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            if np.isnan(loss):
                raise ValueError("loss is nan")

            if (epoch + 1) % args.eval_every != 0:
                print(
                    f"Epoch {epoch+1:03d} [{time.time()-t1:.1f}s] "
                    f"train==[{loss:.5f}={mf_loss:.5f}+{emb_loss:.5f}]"
                )
                continue

            t2 = time.time()
            ret = evaluate_model(sess, model, data, topk_list)
            t3 = time.time()

            print(
                f"Epoch {epoch+1:03d} [{t2-t1:.1f}s + {t3-t2:.1f}s] "
                f"train==[{loss:.5f}={mf_loss:.5f}+{emb_loss:.5f}] "
                f"recall={ret['recall']} precision={ret['precision']} ndcg={ret['ndcg']}"
            )

        print(f"\n[TRAIN DONE] total_time={time.time()-t0:.1f}s")

        user_emb, item_emb = sess.run([model.user_embedding, model.item_embedding])

        user_dict = {int(i): user_emb[i] for i in range(user_emb.shape[0])}
        item_dict = {int(i): item_emb[i] for i in range(item_emb.shape[0])}

        user_path = os.path.join(dataset_dir, "user_pretrain.pk")
        item_path = os.path.join(dataset_dir, "item_pretrain.pk")

        with open(user_path, "wb") as f:
            pkl.dump(user_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

        with open(item_path, "wb") as f:
            pkl.dump(item_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

        print(f"[SAVE] {user_path}")
        print(f"[SAVE] {item_path}")
        print("Saved user_pretrain.pk & item_pretrain.pk")


if __name__ == "__main__":
    main()