import os
import random
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class RecEraserBPRTF:
    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = int(n_users)
        self.n_items = int(n_items)

        self.lr = float(getattr(cfg, "lr", 0.05))
        self.emb_dim = int(getattr(cfg, "emb_dim", 64))
        self.attention_size = max(1, self.emb_dim // 2)
        self.batch_size = int(getattr(cfg, "batch_size", 512))
        self.decay = float(getattr(cfg, "reg_lambda", 1e-5))
        self.num_local = int(getattr(cfg, "shard_num", getattr(cfg, "part_num", 5)))
        self.keep_prob = float(getattr(cfg, "dropout", 0.9))
        self.seed = int(getattr(cfg, "seed", 2024))

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.users = tf.placeholder(tf.int32, shape=(None,), name="users")
            self.pos_items = tf.placeholder(tf.int32, shape=(None,), name="pos_items")
            self.neg_items = tf.placeholder(tf.int32, shape=(None,), name="neg_items")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            self.pred_users = tf.placeholder(tf.int32, shape=(None,), name="pred_users")

            self.weights = self._init_weights()

            self.opt_local = []
            self.loss_local = []
            self.mf_loss_local = []
            self.reg_loss_local = []
            self.batch_ratings_local = []

            for local_id in range(self.num_local):
                opt, loss, mf_loss, reg_loss, batch_ratings = self.train_single_model(local_id)
                self.opt_local.append(opt)
                self.loss_local.append(loss)
                self.mf_loss_local.append(mf_loss)
                self.reg_loss_local.append(reg_loss)
                self.batch_ratings_local.append(batch_ratings)

            (
                self.opt_agg,
                self.loss_agg,
                self.mf_loss_agg,
                self.reg_loss_agg,
                self.attention_loss,
                self.batch_ratings,
                self.u_w,
            ) = self.train_agg_model()

            self.pred_scores = self.build_prediction_graph()
            self.init_op = tf.global_variables_initializer()

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True

        self.sess = tf.Session(graph=self.graph, config=session_config)
        self.sess.run(self.init_op)

    # =========================================================
    # GRAPH / WEIGHTS
    # =========================================================
    def _init_weights(self):
        all_weights = {}
        initializer = tf.glorot_uniform_initializer()

        all_weights["user_embedding"] = tf.Variable(
            initializer([self.n_users, self.num_local, self.emb_dim]),
            name="user_embedding"
        )
        all_weights["item_embedding"] = tf.Variable(
            initializer([self.n_items, self.num_local, self.emb_dim]),
            name="item_embedding"
        )

        stddev = np.sqrt(2.0 / float(self.attention_size + self.emb_dim))

        all_weights["WA"] = tf.Variable(
            tf.random.truncated_normal(
                shape=[self.emb_dim, self.attention_size],
                mean=0.0,
                stddev=stddev
            ),
            dtype=tf.float32,
            name="WA"
        )
        all_weights["BA"] = tf.Variable(
            tf.constant(0.0, shape=[self.attention_size]),
            name="BA"
        )
        all_weights["HA"] = tf.Variable(
            tf.constant(0.01, shape=[self.attention_size, 1]),
            name="HA"
        )

        all_weights["WB"] = tf.Variable(
            tf.random.truncated_normal(
                shape=[self.emb_dim, self.attention_size],
                mean=0.0,
                stddev=stddev
            ),
            dtype=tf.float32,
            name="WB"
        )
        all_weights["BB"] = tf.Variable(
            tf.constant(0.0, shape=[self.attention_size]),
            name="BB"
        )
        all_weights["HB"] = tf.Variable(
            tf.constant(0.01, shape=[self.attention_size, 1]),
            name="HB"
        )

        all_weights["trans_W"] = tf.Variable(
            initializer([self.num_local, self.emb_dim, self.emb_dim]),
            name="trans_W"
        )
        all_weights["trans_B"] = tf.Variable(
            initializer([self.num_local, self.emb_dim]),
            name="trans_B"
        )
        return all_weights

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
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    # =========================================================
    # LOCAL MODEL
    # =========================================================
    def train_single_model(self, local_num):
        u_e = tf.nn.embedding_lookup(self.weights["user_embedding"][:, local_num], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights["item_embedding"][:, local_num], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights["item_embedding"][:, local_num], self.neg_items)

        mf_loss, reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        loss = mf_loss + reg_loss

        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        opt = tf.train.AdagradOptimizer(
            learning_rate=self.lr,
            initial_accumulator_value=1e-8
        ).minimize(loss)

        return opt, loss, mf_loss, reg_loss, batch_ratings

    # =========================================================
    # ATTENTION AGG
    # =========================================================
    def attention_based_agg(self, embs, flag):
        if flag == 0:
            embs_w = tf.exp(
                tf.einsum(
                    "abc,ck->abk",
                    tf.nn.relu(tf.einsum("abc,ck->abk", embs, self.weights["WA"]) + self.weights["BA"]),
                    self.weights["HA"]
                )
            )
        else:
            embs_w = tf.exp(
                tf.einsum(
                    "abc,ck->abk",
                    tf.nn.relu(tf.einsum("abc,ck->abk", embs, self.weights["WB"]) + self.weights["BB"]),
                    self.weights["HB"]
                )
            )

        embs_w = tf.divide(embs_w, tf.reduce_sum(embs_w, 1, keepdims=True) + 1e-10)
        agg_emb = tf.reduce_sum(tf.multiply(embs_w, embs), 1)
        return agg_emb, embs_w

    def train_agg_model(self):
        u_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights["user_embedding"], self.users))
        pos_i_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights["item_embedding"], self.pos_items))
        neg_i_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights["item_embedding"], self.neg_items))

        u_es = tf.einsum("abc,bcd->abd", u_es, self.weights["trans_W"]) + self.weights["trans_B"]
        pos_i_es = tf.einsum("abc,bcd->abd", pos_i_es, self.weights["trans_W"]) + self.weights["trans_B"]
        neg_i_es = tf.einsum("abc,bcd->abd", neg_i_es, self.weights["trans_W"]) + self.weights["trans_B"]

        u_e, u_w = self.attention_based_agg(u_es, 0)
        pos_i_e, _ = self.attention_based_agg(pos_i_es, 1)
        neg_i_e, _ = self.attention_based_agg(neg_i_es, 1)

        u_e_drop = tf.nn.dropout(u_e, keep_prob=self.dropout_keep_prob)

        mf_loss, reg_loss_bpr = self.create_bpr_loss(u_e_drop, pos_i_e, neg_i_e)

        l2_loss = self.decay * (
            tf.nn.l2_loss(self.weights["WA"]) +
            tf.nn.l2_loss(self.weights["BA"]) +
            tf.nn.l2_loss(self.weights["HA"]) +
            tf.nn.l2_loss(self.weights["WB"]) +
            tf.nn.l2_loss(self.weights["BB"]) +
            tf.nn.l2_loss(self.weights["HB"])
        )

        reg_loss = 1e-5 * (
            tf.nn.l2_loss(self.weights["trans_W"]) +
            tf.nn.l2_loss(self.weights["trans_B"])
        )

        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)
        loss = mf_loss + reg_loss_bpr + reg_loss + l2_loss

        opt = tf.train.AdagradOptimizer(
            learning_rate=self.lr,
            initial_accumulator_value=1e-8
        ).minimize(loss)

        return opt, loss, mf_loss, reg_loss, l2_loss, batch_ratings, u_w

    # =========================================================
    # PREDICTION
    # =========================================================
    def build_prediction_graph(self):
        pred_user_local = tf.nn.embedding_lookup(self.weights["user_embedding"], self.pred_users)
        pred_user_local = tf.einsum("abc,bcd->abd", pred_user_local, self.weights["trans_W"]) + self.weights["trans_B"]
        pred_user_agg, _ = self.attention_based_agg(pred_user_local, 0)

        all_item_local = self.weights["item_embedding"]
        all_item_local = tf.einsum("abc,bcd->abd", all_item_local, self.weights["trans_W"]) + self.weights["trans_B"]
        all_item_agg, _ = self.attention_based_agg(all_item_local, 1)

        scores = tf.matmul(pred_user_agg, all_item_agg, transpose_b=True)
        return scores

    # =========================================================
    # TRAIN API
    # =========================================================
    def fit_local_epoch(self, loader, local_id):
        n_batch = max(1, loader.n_C[local_id] // self.batch_size + 1)
        loss = mf_loss = reg_loss = 0.0
        eff_batch = 0

        for _ in range(n_batch):
            users, pos_items, neg_items = loader.local_sample(local_id)
            if len(users) == 0:
                continue

            _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
                [
                    self.opt_local[local_id],
                    self.loss_local[local_id],
                    self.mf_loss_local[local_id],
                    self.reg_loss_local[local_id],
                ],
                feed_dict={
                    self.users: users,
                    self.pos_items: pos_items,
                    self.neg_items: neg_items,
                    self.dropout_keep_prob: self.keep_prob,
                },
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
            eff_batch += 1

        if eff_batch == 0:
            return 0.0, 0.0, 0.0

        return float(loss / eff_batch), float(mf_loss / eff_batch), float(reg_loss / eff_batch)

    def fit_agg_epoch(self, loader):
        n_batch = max(1, loader.n_train // self.batch_size + 1)
        loss = mf_loss = reg_loss = att_loss = 0.0
        eff_batch = 0

        for _ in range(n_batch):
            users, pos_items, neg_items = loader.sample()
            if len(users) == 0:
                continue

            _, batch_loss, batch_mf_loss, batch_reg_loss, batch_attention_loss = self.sess.run(
                [
                    self.opt_agg,
                    self.loss_agg,
                    self.mf_loss_agg,
                    self.reg_loss_agg,
                    self.attention_loss,
                ],
                feed_dict={
                    self.users: users,
                    self.pos_items: pos_items,
                    self.neg_items: neg_items,
                    self.dropout_keep_prob: self.keep_prob,
                },
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
            att_loss += batch_attention_loss
            eff_batch += 1

        if eff_batch == 0:
            return 0.0, 0.0, 0.0, 0.0

        return (
            float(loss / eff_batch),
            float(mf_loss / eff_batch),
            float(reg_loss / eff_batch),
            float(att_loss / eff_batch),
        )

    def predict(self, user_id):
        scores = self.sess.run(
            self.pred_scores,
            feed_dict={
                self.pred_users: np.asarray([int(user_id)], dtype=np.int32),
            },
        )
        return scores[0]

    # =========================================================
    # STATE
    # =========================================================
    def get_state(self):
        with self.graph.as_default():
            values = self.sess.run(tf.global_variables())
            names = [v.name for v in tf.global_variables()]
        return {name: value for name, value in zip(names, values)}

    def set_state(self, state):
        with self.graph.as_default():
            assigns = []
            for var in tf.global_variables():
                if var.name in state:
                    assigns.append(tf.assign(var, state[var.name]))
            if assigns:
                self.sess.run(assigns)

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None


class RecEraserBPRWrapper:
    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.model = RecEraserBPRTF(cfg, n_users, n_items)

    def fit_local(self, loader, local_id, epochs=None):
        if epochs is None:
            epochs = getattr(self.cfg, "local_epochs", getattr(self.cfg, "epochs", 1))

        epochs = max(1, int(epochs))
        last_stats = None

        for epoch in range(epochs):
            loss, mf_loss, reg_loss = self.model.fit_local_epoch(loader, local_id)
            last_stats = {
                "loss": loss,
                "mf_loss": mf_loss,
                "reg_loss": reg_loss,
                "local_id": local_id,
                "epoch": epoch + 1,
            }

            if getattr(self.cfg, "print_loss", False):
                print(
                    f"[REC LOCAL][shard={local_id}][Epoch {epoch+1}/{epochs}] "
                    f"loss={loss:.6f}, mf={mf_loss:.6f}, reg={reg_loss:.6f}"
                )

        return last_stats

    def fit_agg(self, loader, epochs=None):
        if epochs is None:
            epochs = getattr(self.cfg, "epoch_agg", getattr(self.cfg, "agg_epochs", 1))

        epochs = max(1, int(epochs))
        last_stats = None

        for epoch in range(epochs):
            loss, mf_loss, reg_loss, attention_loss = self.model.fit_agg_epoch(loader)
            last_stats = {
                "loss": loss,
                "mf_loss": mf_loss,
                "reg_loss": reg_loss,
                "attention_loss": attention_loss,
                "epoch": epoch + 1,
            }

            if getattr(self.cfg, "print_loss", False):
                print(
                    f"[REC AGG][Epoch {epoch+1}/{epochs}] "
                    f"loss={loss:.6f}, mf={mf_loss:.6f}, reg={reg_loss:.6f}, "
                    f"attention={attention_loss:.6f}"
                )

        return last_stats

    def fit(self, *args, **kwargs):
        raise NotImplementedError("RecEraser uses fit_local() and fit_agg(), not fit().")

    def predict(self, user_id):
        return self.model.predict(user_id)

    def get_state(self):
        return self.model.get_state()

    def set_state(self, state):
        self.model.set_state(state)

    def clone_fresh(self):
        return RecEraserBPRWrapper(self.cfg, self.n_users, self.n_items)

    def close(self):
        self.model.close()