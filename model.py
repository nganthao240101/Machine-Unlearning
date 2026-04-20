import tensorflow as tf


def to_scalar(x):
    try:
        return float(x.numpy())
    except Exception:
        pass

    try:
        return float(tf.keras.backend.get_value(x))
    except Exception:
        pass

    try:
        sess = tf.compat.v1.keras.backend.get_session()
        return float(sess.run(x))
    except Exception:
        pass

    return float(x)


class BPRModel:
    def __init__(self, n_users, n_items, emb_dim, lr, reg_lambda=1e-4):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.lr = lr
        self.reg_lambda = reg_lambda

        self.user_emb = tf.keras.layers.Embedding(
            input_dim=n_users,
            output_dim=emb_dim,
            embeddings_initializer="random_normal",
            name="user_embedding"
        )

        self.item_emb = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=emb_dim,
            embeddings_initializer="random_normal",
            name="item_embedding"
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # build
        self.user_emb(tf.convert_to_tensor([0], dtype=tf.int32))
        self.item_emb(tf.convert_to_tensor([0], dtype=tf.int32))

    def compute_scores(self, user_vecs, item_vecs):
        return tf.reduce_sum(user_vecs * item_vecs, axis=1)

    def bpr_loss(self, users, pos_items, neg_items):
        users = tf.convert_to_tensor(users, dtype=tf.int32)
        pos_items = tf.convert_to_tensor(pos_items, dtype=tf.int32)
        neg_items = tf.convert_to_tensor(neg_items, dtype=tf.int32)

        u = self.user_emb(users)
        p = self.item_emb(pos_items)
        n = self.item_emb(neg_items)

        pos_scores = self.compute_scores(u, p)
        neg_scores = self.compute_scores(u, n)

        ranking_loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))
        reg_loss = self.reg_lambda * (
            tf.nn.l2_loss(u) + tf.nn.l2_loss(p) + tf.nn.l2_loss(n)
        ) / tf.cast(tf.shape(u)[0], tf.float32)

        total_loss = ranking_loss + reg_loss
        return total_loss, ranking_loss, reg_loss

    def train_step(self, users, pos_items, neg_items):
        users = tf.convert_to_tensor(users, dtype=tf.int32)
        pos_items = tf.convert_to_tensor(pos_items, dtype=tf.int32)
        neg_items = tf.convert_to_tensor(neg_items, dtype=tf.int32)

        with tf.GradientTape() as tape:
            total_loss, ranking_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)

        variables = self.user_emb.trainable_variables + self.item_emb.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return to_scalar(total_loss), to_scalar(ranking_loss), to_scalar(reg_loss)

    def predict(self, user_id):
        user_id = tf.convert_to_tensor([user_id], dtype=tf.int32)
        u = self.user_emb(user_id)
        all_items = self.item_emb.weights[0]
        scores = tf.matmul(u, all_items, transpose_b=True)

        try:
            return scores.numpy().flatten()
        except Exception:
            try:
                return tf.keras.backend.get_value(scores).flatten()
            except Exception:
                sess = tf.compat.v1.keras.backend.get_session()
                return sess.run(scores).flatten()

    def get_weights_dict(self):
        return {
            "user_emb": self.user_emb.get_weights(),
            "item_emb": self.item_emb.get_weights()
        }

    def set_weights_dict(self, weights):
        if "user_emb" in weights and "item_emb" in weights:
            self.user_emb.set_weights(weights["user_emb"])
            self.item_emb.set_weights(weights["item_emb"])
        elif "u" in weights and "i" in weights:
            self.user_emb.set_weights(weights["u"])
            self.item_emb.set_weights(weights["i"])
        else:
            raise ValueError("Unknown checkpoint format")