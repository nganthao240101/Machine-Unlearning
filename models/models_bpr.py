import random
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from models.models_base import BaseModel


def sample_bpr_batch(train_user_dict, n_items, batch_size):
    users, pos, neg = [], [], []
    user_list = [u for u, items in train_user_dict.items() if len(items) > 0]

    if len(user_list) == 0:
        return np.array([]), np.array([]), np.array([])

    tries = 0
    max_tries = batch_size * 10

    while len(users) < batch_size and tries < max_tries:
        tries += 1
        u = random.choice(user_list)
        pos_items = train_user_dict[u]

        if not pos_items:
            continue

        if len(pos_items) >= n_items:
            continue

        p = random.choice(pos_items)

        guard = 0
        n = random.randint(0, n_items - 1)
        while n in pos_items and guard < max(100, n_items * 2):
            n = random.randint(0, n_items - 1)
            guard += 1

        if n in pos_items:
            continue

        users.append(u)
        pos.append(p)
        neg.append(n)

    return (
        np.array(users, dtype=np.int32),
        np.array(pos, dtype=np.int32),
        np.array(neg, dtype=np.int32),
    )


class BPRTF:
    def __init__(self, cfg, n_users, n_items, emb_dim, lr, reg_lambda):
        self.cfg = cfg
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.seed = int(getattr(cfg, "seed", 2024))

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.users = tf.placeholder(tf.int32, shape=[None], name="users")
            self.pos_items = tf.placeholder(tf.int32, shape=[None], name="pos_items")
            self.neg_items = tf.placeholder(tf.int32, shape=[None], name="neg_items")

            initializer = tf.random_normal_initializer(stddev=0.01)

            self.user_emb = tf.get_variable(
                "user_emb",
                shape=[n_users, emb_dim],
                initializer=initializer
            )
            self.item_emb = tf.get_variable(
                "item_emb",
                shape=[n_items, emb_dim],
                initializer=initializer
            )

            u = tf.nn.embedding_lookup(self.user_emb, self.users)
            p = tf.nn.embedding_lookup(self.item_emb, self.pos_items)
            n = tf.nn.embedding_lookup(self.item_emb, self.neg_items)

            pos_scores = tf.reduce_sum(u * p, axis=1)
            neg_scores = tf.reduce_sum(u * n, axis=1)

            self.mf_loss = -tf.reduce_mean(
                tf.log(tf.nn.sigmoid(pos_scores - neg_scores) + 1e-10)
            )

            batch_size_tensor = tf.cast(tf.shape(u)[0], tf.float32)
            self.reg_loss = self.reg_lambda * (
                tf.nn.l2_loss(u) + tf.nn.l2_loss(p) + tf.nn.l2_loss(n)
            ) / batch_size_tensor

            self.loss = self.mf_loss + self.reg_loss
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.all_item_scores = tf.matmul(u, self.item_emb, transpose_b=True)
            self.init_op = tf.global_variables_initializer()

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True
        self.sess = tf.Session(graph=self.graph, config=session_config)
        self.sess.run(self.init_op)

    def train_step(self, users, pos_items, neg_items):
        with self.graph.as_default():
            feed_dict = {
                self.users: users,
                self.pos_items: pos_items,
                self.neg_items: neg_items,
            }

            _, loss, mf_loss, reg_loss = self.sess.run(
                [self.train_op, self.loss, self.mf_loss, self.reg_loss],
                feed_dict=feed_dict
            )

        return float(loss), float(mf_loss), float(reg_loss)

    def predict(self, user_id):
        with self.graph.as_default():
            scores = self.sess.run(
                self.all_item_scores,
                feed_dict={
                    self.users: np.array([user_id], dtype=np.int32)
                }
            )
        return scores.flatten()

    def get_state(self):
        with self.graph.as_default():
            user_emb, item_emb = self.sess.run([self.user_emb, self.item_emb])

        return {
            "user_emb": user_emb,
            "item_emb": item_emb,
        }

    def set_state(self, state):
        with self.graph.as_default():
            assign_user = tf.assign(self.user_emb, state["user_emb"])
            assign_item = tf.assign(self.item_emb, state["item_emb"])
            self.sess.run([assign_user, assign_item])

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None


class BPRWrapper(BaseModel):
    def __init__(self, cfg, n_users, n_items):
        super().__init__(cfg, n_users, n_items)
        self.model = BPRTF(
            cfg=cfg,
            n_users=n_users,
            n_items=n_items,
            emb_dim=cfg.emb_dim,
            lr=cfg.lr,
            reg_lambda=cfg.reg_lambda
        )

    def _resolve_train_user_dict(self, train_data):
        if isinstance(train_data, dict):
            return train_data

        for attr in ["train_user_dict", "train_dict", "user_dict"]:
            if hasattr(train_data, attr):
                value = getattr(train_data, attr)
                if isinstance(value, dict):
                    return value

        raise TypeError(
            "BPRWrapper.fit() expects either:\n"
            "  - a dict like {user: [items]}\n"
            "  - or an object with one of attributes: "
            "train_user_dict / train_dict / user_dict\n"
            f"but got: {type(train_data)}"
        )

    def fit_one_epoch(self, train_user_dict):
        total, mf, reg, steps = 0.0, 0.0, 0.0, 0

        n_interactions = sum(len(items) for items in train_user_dict.values())

        if n_interactions == 0:
            return {
                "loss": 0.0,
                "mf_loss": 0.0,
                "reg_loss": 0.0,
                "steps": 0,
                "n_interactions": 0
            }

        n_batches = max(1, int(np.ceil(n_interactions / self.cfg.batch_size)))

        for _ in range(n_batches):
            u, p, n = sample_bpr_batch(
                train_user_dict=train_user_dict,
                n_items=self.n_items,
                batch_size=self.cfg.batch_size
            )

            if len(u) == 0:
                continue

            l, lm, lr = self.model.train_step(u, p, n)
            total += l
            mf += lm
            reg += lr
            steps += 1

        if steps == 0:
            return {
                "loss": 0.0,
                "mf_loss": 0.0,
                "reg_loss": 0.0,
                "steps": 0,
                "n_interactions": n_interactions
            }

        return {
            "loss": total / steps,
            "mf_loss": mf / steps,
            "reg_loss": reg / steps,
            "steps": steps,
            "n_interactions": n_interactions
        }

    def fit(self, train_data, epochs=None):
        train_user_dict = self._resolve_train_user_dict(train_data)

        if epochs is None:
            epochs = self.cfg.epochs

        n_users_nonempty = sum(1 for _, items in train_user_dict.items() if len(items) > 0)
        n_interactions = sum(len(items) for items in train_user_dict.values())

        print(f"[BPR] users_with_items={n_users_nonempty}, interactions={n_interactions}")

        if n_interactions == 0:
            print("[BPR] Empty shard/local_data -> skip training")
            return {
                "loss": 0.0,
                "mf_loss": 0.0,
                "reg_loss": 0.0,
                "steps": 0,
                "n_interactions": 0
            }

        last_stats = None
        for epoch in range(epochs):
            last_stats = self.fit_one_epoch(train_user_dict)

            if getattr(self.cfg, "print_loss", False):
                print(
                    f"[BPR][Epoch {epoch+1}/{epochs}] "
                    f"loss={last_stats['loss']:.6f}, "
                    f"mf={last_stats['mf_loss']:.6f}, "
                    f"reg={last_stats['reg_loss']:.6f}, "
                    f"steps={last_stats.get('steps', -1)}, "
                    f"interactions={last_stats.get('n_interactions', -1)}"
                )

        return last_stats

    def predict(self, user_id):
        return self.model.predict(user_id)

    def clone_fresh(self):
        return BPRWrapper(self.cfg, self.n_users, self.n_items)

    def get_state(self):
        return self.model.get_state()

    def set_state(self, state):
        self.model.set_state(state)

    def close(self):
        self.model.close()