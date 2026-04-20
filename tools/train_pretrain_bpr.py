import os
import time
import pickle as pkl
import random
import argparse

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train BPR pretrain embeddings for RecEraser")

    parser.add_argument("--dataset", type=str, default="ml-1m",
                        help="Dataset name: ml-1m | ml-10m | yelp2018")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (paper-like)")
    parser.add_argument("--embed_size", type=int, default=64,
                        help="Embedding size")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--reg", type=float, default=1e-3,
                        help="L2 regularization")
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")
    parser.add_argument("--topk", type=str, default="10,20,50",
                        help="Top-K list, e.g. 10,20,50")

    return parser.parse_args()


# =========================================================
# DATA LOADER
# =========================================================
class BPRData:
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

        self._load()

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
class BPRMF:
    def __init__(self, n_users, n_items, emb_dim=64, lr=0.001, reg=1e-3):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.reg = reg

        self.users = tf.placeholder(tf.int32, shape=(None,), name="users")
        self.pos_items = tf.placeholder(tf.int32, shape=(None,), name="pos_items")
        self.neg_items = tf.placeholder(tf.int32, shape=(None,), name="neg_items")

        initializer = tf.glorot_uniform_initializer()

        self.user_embedding = tf.Variable(
            initializer([self.n_users, self.emb_dim]),
            name="user_embedding"
        )
        self.item_embedding = tf.Variable(
            initializer([self.n_items, self.emb_dim]),
            name="item_embedding"
        )

        u_e = tf.nn.embedding_lookup(self.user_embedding, self.users)
        pos_i_e = tf.nn.embedding_lookup(self.item_embedding, self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.item_embedding, self.neg_items)

        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        self.mf_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.reg_loss

        self.opt = tf.train.AdagradOptimizer(
            learning_rate=self.lr,
            initial_accumulator_value=1e-8
        ).minimize(self.loss)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(users) +
            tf.nn.l2_loss(pos_items) +
            tf.nn.l2_loss(neg_items)
        )
        regularizer = regularizer / tf.cast(tf.maximum(tf.shape(users)[0], 1), tf.float32)

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.reg * regularizer
        return mf_loss, reg_loss


# =========================================================
# EVALUATION
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

        # mask seen train items
        for seen_i in data.train_user_dict.get(u, []):
            scores = scores.copy()
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
# MAIN TRAIN
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

    data = BPRData(train_path, test_path, seed=args.seed)

    model = BPRMF(
        n_users=data.n_users,
        n_items=data.n_items,
        emb_dim=args.embed_size,
        lr=args.lr,
        reg=args.reg,
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    t0 = time.time()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("\n[TRAIN BPR PRETRAIN]")
        print(f"  dataset     : {args.dataset}")
        print(f"  epochs      : {args.epochs}")
        print(f"  emb_dim     : {args.embed_size}")
        print(f"  batch_size  : {args.batch_size}")
        print(f"  lr          : {args.lr}")
        print(f"  reg         : {args.reg}")
        print(f"  topk        : {topk_list}")

        for epoch in range(args.epochs):
            t1 = time.time()
            loss, mf_loss, reg_loss = 0.0, 0.0, 0.0

            n_batch = data.n_train // args.batch_size + 1

            for _ in range(n_batch):
                users, pos_items, neg_items = data.sample(args.batch_size)

                _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                    [model.opt, model.loss, model.mf_loss, model.reg_loss],
                    feed_dict={
                        model.users: users,
                        model.pos_items: pos_items,
                        model.neg_items: neg_items,
                    }
                )

                loss += batch_loss
                mf_loss += batch_mf_loss
                reg_loss += batch_reg_loss

            if np.isnan(loss):
                raise ValueError("loss is nan")

            if (epoch + 1) % args.eval_every != 0:
                print(
                    f"Epoch {epoch+1:03d} [{time.time()-t1:.1f}s] "
                    f"train==[{loss:.5f}={mf_loss:.5f}+{reg_loss:.5f}]"
                )
                continue

            t2 = time.time()
            ret = evaluate_model(sess, model, data, topk_list)
            t3 = time.time()

            print(
                f"Epoch {epoch+1:03d} [{t2-t1:.1f}s + {t3-t2:.1f}s] "
                f"train==[{loss:.5f}={mf_loss:.5f}+{reg_loss:.5f}] "
                f"recall={ret['recall']} precision={ret['precision']} ndcg={ret['ndcg']}"
            )

        print(f"\n[TRAIN DONE] total_time={time.time()-t0:.1f}s")

        # save embeddings
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