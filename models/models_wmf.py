import random
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from models.models_base import BaseModel


def sample_wmf_pairs(train_user_dict, n_items, batch_size, neg_ratio=3):
    users, items, labels = [], [], []

    user_list = [u for u, v in train_user_dict.items() if len(v) > 0]
    if not user_list:
        return np.array([]), np.array([]), np.array([])

    tries = 0
    max_tries = batch_size * 10

    while len(users) < batch_size and tries < max_tries:
        tries += 1

        u = random.choice(user_list)
        pos_items = train_user_dict[u]

        if not pos_items:
            continue

        p = random.choice(pos_items)
        users.append(u)
        items.append(p)
        labels.append(1.0)

        for _ in range(neg_ratio):
            n = random.randint(0, n_items - 1)
            while n in pos_items:
                n = random.randint(0, n_items - 1)

            users.append(u)
            items.append(n)
            labels.append(0.0)

    return (
        np.array(users, dtype=np.int32),
        np.array(items, dtype=np.int32),
        np.array(labels, dtype=np.float32),
    )


class WMFTF:
    def __init__(self, n_users, n_items, emb_dim, lr, reg_lambda):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.reg_lambda = reg_lambda

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.users = tf.placeholder(tf.int32, shape=[None], name="users")
            self.items = tf.placeholder(tf.int32, shape=[None], name="items")
            self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

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
            i = tf.nn.embedding_lookup(self.item_emb, self.items)

            scores = tf.reduce_sum(u * i, axis=1)
            pred = tf.nn.sigmoid(scores)

            weights = 1.0 + 4.0 * self.labels
            self.mf_loss = tf.reduce_mean(weights * tf.square(self.labels - pred))

            batch_size_tensor = tf.cast(tf.shape(u)[0], tf.float32)
            self.reg_loss = self.reg_lambda * (
                tf.nn.l2_loss(u) + tf.nn.l2_loss(i)
            ) / batch_size_tensor

            self.loss = self.mf_loss + self.reg_loss
            self.train_op = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)

            self.single_user_id = tf.placeholder(tf.int32, shape=[1], name="single_user_id")
            single_u = tf.nn.embedding_lookup(self.user_emb, self.single_user_id)
            self.single_user_scores = tf.matmul(single_u, self.item_emb, transpose_b=True)

            self.init_op = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def train_step(self, users, items, labels):
        with self.graph.as_default():
            feed_dict = {
                self.users: users,
                self.items: items,
                self.labels: labels,
            }

            _, loss, mf_loss, reg_loss = self.sess.run(
                [self.train_op, self.loss, self.mf_loss, self.reg_loss],
                feed_dict=feed_dict
            )

        return float(loss), float(mf_loss), float(reg_loss)

    def predict(self, user_id):
        with self.graph.as_default():
            scores = self.sess.run(
                self.single_user_scores,
                feed_dict={
                    self.single_user_id: np.array([user_id], dtype=np.int32)
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


class WMFWrapper(BaseModel):
    def __init__(self, cfg, n_users, n_items):
        super().__init__(cfg, n_users, n_items)
        self.model = WMFTF(
            n_users=n_users,
            n_items=n_items,
            emb_dim=cfg.emb_dim,
            lr=cfg.lr,
            reg_lambda=cfg.reg_lambda
        )

    def fit_one_epoch(self, train_user_dict):
        total, mf, reg, steps = 0.0, 0.0, 0.0, 0

        for _ in range(50):
            u, i, y = sample_wmf_pairs(
                train_user_dict=train_user_dict,
                n_items=self.n_items,
                batch_size=self.cfg.batch_size
            )

            if len(u) == 0:
                continue

            l, lm, lr = self.model.train_step(u, i, y)
            total += l
            mf += lm
            reg += lr
            steps += 1

        if steps == 0:
            return {
                "loss": 0.0,
                "mf_loss": 0.0,
                "reg_loss": 0.0
            }

        return {
            "loss": total / steps,
            "mf_loss": mf / steps,
            "reg_loss": reg / steps
        }

    def fit(self, train_user_dict, epochs=None):
        if epochs is None:
            epochs = self.cfg.epochs

        last_stats = None
        for epoch in range(epochs):
            last_stats = self.fit_one_epoch(train_user_dict)

            if getattr(self.cfg, "print_loss", False):
                print(
                    f"[WMF][Epoch {epoch+1}/{epochs}] "
                    f"loss={last_stats['loss']:.6f}, "
                    f"mf={last_stats['mf_loss']:.6f}, "
                    f"reg={last_stats['reg_loss']:.6f}"
                )

        return last_stats

    def predict(self, user_id):
        return self.model.predict(user_id)

    def clone_fresh(self):
        return WMFWrapper(self.cfg, self.n_users, self.n_items)

    def get_state(self):
        return self.model.get_state()

    def set_state(self, state):
        self.model.set_state(state)

    def state_dict(self):
        return self.get_state()

    def load_state_dict(self, state):
        self.set_state(state)

    def close(self):
        self.model.close()