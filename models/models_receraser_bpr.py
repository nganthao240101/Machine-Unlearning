
import os
import random
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class RecEraserBPRTF:
    """
    TensorFlow 1.x implementation of RecEraser-BPR with:
    - local shard-specific BPR training
    - trainable aggregation module (transform + attention)
    - prediction on aggregated user/item embeddings

    Main fixes compared with many rough drafts:
    1) local optimization updates only the corresponding shard slice
    2) aggregation optimization updates only aggregation parameters
    3) local loss and agg loss are separated clearly
    4) prediction uses the same aggregation path as training
    5) helper methods added for loading local embeddings and cached states
    """

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
            self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout_keep_prob")
            self.pred_users = tf.placeholder(tf.int32, shape=(None,), name="pred_users")

            self.weights = self._init_weights()

            # local train ops / losses
            self.opt_local = []
            self.loss_local = []
            self.mf_loss_local = []
            self.reg_loss_local = []

            for local_id in range(self.num_local):
                opt, loss, mf_loss, reg_loss = self._build_local_train_graph(local_id)
                self.opt_local.append(opt)
                self.loss_local.append(loss)
                self.mf_loss_local.append(mf_loss)
                self.reg_loss_local.append(reg_loss)

            (
                self.opt_agg,
                self.loss_agg,
                self.mf_loss_agg,
                self.reg_loss_agg,
                self.attention_loss,
                self.pred_train_scores,
                self.u_w,
            ) = self._build_agg_train_graph()

            self.pred_scores = self._build_prediction_graph()
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
        initializer = tf.glorot_uniform_initializer(seed=self.seed)
        all_weights = {}

        # Keep local embeddings as separate variables to make shard updates precise.
        all_weights["user_embedding_list"] = [
            tf.Variable(
                initializer([self.n_users, self.emb_dim]),
                name=f"user_embedding_{i}"
            )
            for i in range(self.num_local)
        ]
        all_weights["item_embedding_list"] = [
            tf.Variable(
                initializer([self.n_items, self.emb_dim]),
                name=f"item_embedding_{i}"
            )
            for i in range(self.num_local)
        ]

        stddev = np.sqrt(2.0 / float(self.attention_size + self.emb_dim))

        all_weights["WA"] = tf.Variable(
            tf.random.truncated_normal(
                shape=[self.emb_dim, self.attention_size],
                mean=0.0,
                stddev=stddev,
                seed=self.seed,
            ),
            dtype=tf.float32,
            name="WA",
        )
        all_weights["BA"] = tf.Variable(
            tf.zeros([self.attention_size], dtype=tf.float32),
            name="BA",
        )
        all_weights["HA"] = tf.Variable(
            tf.constant(0.01, shape=[self.attention_size, 1], dtype=tf.float32),
            name="HA",
        )

        all_weights["WB"] = tf.Variable(
            tf.random.truncated_normal(
                shape=[self.emb_dim, self.attention_size],
                mean=0.0,
                stddev=stddev,
                seed=self.seed + 1,
            ),
            dtype=tf.float32,
            name="WB",
        )
        all_weights["BB"] = tf.Variable(
            tf.zeros([self.attention_size], dtype=tf.float32),
            name="BB",
        )
        all_weights["HB"] = tf.Variable(
            tf.constant(0.01, shape=[self.attention_size, 1], dtype=tf.float32),
            name="HB",
        )

        all_weights["trans_W"] = tf.Variable(
            initializer([self.num_local, self.emb_dim, self.emb_dim]),
            name="trans_W",
        )
        all_weights["trans_B"] = tf.Variable(
            initializer([self.num_local, self.emb_dim]),
            name="trans_B",
        )
        return all_weights

    def _stack_user_embeddings(self):
        return tf.stack(self.weights["user_embedding_list"], axis=1)

    def _stack_item_embeddings(self):
        return tf.stack(self.weights["item_embedding_list"], axis=1)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(users * pos_items, axis=1)
        neg_scores = tf.reduce_sum(users * neg_items, axis=1)

        regularizer = (
            tf.nn.l2_loss(users) +
            tf.nn.l2_loss(pos_items) +
            tf.nn.l2_loss(neg_items)
        )
        batch_den = tf.cast(tf.maximum(tf.shape(users)[0], 1), tf.float32)
        regularizer = regularizer / batch_den

        # numerically safer than log(sigmoid(x))
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    # =========================================================
    # LOCAL MODEL
    # =========================================================
    def _build_local_train_graph(self, local_num):
        user_var = self.weights["user_embedding_list"][local_num]
        item_var = self.weights["item_embedding_list"][local_num]

        u_e = tf.nn.embedding_lookup(user_var, self.users)
        pos_i_e = tf.nn.embedding_lookup(item_var, self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(item_var, self.neg_items)

        mf_loss, reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        loss = mf_loss + reg_loss

        optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.lr,
            initial_accumulator_value=1e-8,
        )
        opt = optimizer.minimize(loss, var_list=[user_var, item_var])

        return opt, loss, mf_loss, reg_loss

    # =========================================================
    # ATTENTION AGG
    # =========================================================
    def attention_based_agg(self, embs, flag):
        if flag == 0:
            hidden = tf.nn.relu(tf.einsum("abc,ck->abk", embs, self.weights["WA"]) + self.weights["BA"])
            logits = tf.einsum("abk,kl->abl", hidden, self.weights["HA"])
        else:
            hidden = tf.nn.relu(tf.einsum("abc,ck->abk", embs, self.weights["WB"]) + self.weights["BB"])
            logits = tf.einsum("abk,kl->abl", hidden, self.weights["HB"])

        weights = tf.nn.softmax(logits, axis=1)
        agg_emb = tf.reduce_sum(weights * embs, axis=1)
        return agg_emb, weights

    def _transform_local_embs(self, embs):
        # embs: [batch_or_items, num_local, emb_dim]
        return tf.einsum("abc,bcd->abd", embs, self.weights["trans_W"]) + self.weights["trans_B"]

    def _build_agg_train_graph(self):
        user_embs = self._stack_user_embeddings()
        item_embs = self._stack_item_embeddings()

        # freeze local embeddings during agg training
        u_es = tf.stop_gradient(tf.nn.embedding_lookup(user_embs, self.users))
        pos_i_es = tf.stop_gradient(tf.nn.embedding_lookup(item_embs, self.pos_items))
        neg_i_es = tf.stop_gradient(tf.nn.embedding_lookup(item_embs, self.neg_items))

        u_es = self._transform_local_embs(u_es)
        pos_i_es = self._transform_local_embs(pos_i_es)
        neg_i_es = self._transform_local_embs(neg_i_es)

        u_e, u_w = self.attention_based_agg(u_es, flag=0)
        pos_i_e, _ = self.attention_based_agg(pos_i_es, flag=1)
        neg_i_e, _ = self.attention_based_agg(neg_i_es, flag=1)

        u_e_drop = tf.nn.dropout(u_e, keep_prob=self.dropout_keep_prob)

        mf_loss, reg_loss_bpr = self.create_bpr_loss(u_e_drop, pos_i_e, neg_i_e)

        attention_reg = self.decay * (
            tf.nn.l2_loss(self.weights["WA"]) +
            tf.nn.l2_loss(self.weights["BA"]) +
            tf.nn.l2_loss(self.weights["HA"]) +
            tf.nn.l2_loss(self.weights["WB"]) +
            tf.nn.l2_loss(self.weights["BB"]) +
            tf.nn.l2_loss(self.weights["HB"])
        )

        trans_reg = 1e-5 * (
            tf.nn.l2_loss(self.weights["trans_W"]) +
            tf.nn.l2_loss(self.weights["trans_B"])
        )

        total_reg = reg_loss_bpr + attention_reg + trans_reg
        loss = mf_loss + total_reg

        train_scores = tf.reduce_sum(u_e * pos_i_e, axis=1)

        agg_vars = [
            self.weights["WA"], self.weights["BA"], self.weights["HA"],
            self.weights["WB"], self.weights["BB"], self.weights["HB"],
            self.weights["trans_W"], self.weights["trans_B"],
        ]
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.lr,
            initial_accumulator_value=1e-8,
        )
        opt = optimizer.minimize(loss, var_list=agg_vars)

        return opt, loss, mf_loss, total_reg, attention_reg, train_scores, u_w

    # =========================================================
    # PREDICTION
    # =========================================================
    def _build_prediction_graph(self):
        user_embs = self._stack_user_embeddings()
        item_embs = self._stack_item_embeddings()

        pred_user_local = tf.nn.embedding_lookup(user_embs, self.pred_users)
        pred_user_local = self._transform_local_embs(pred_user_local)
        pred_user_agg, _ = self.attention_based_agg(pred_user_local, flag=0)

        all_item_local = self._transform_local_embs(item_embs)
        all_item_agg, _ = self.attention_based_agg(all_item_local, flag=1)

        scores = tf.matmul(pred_user_agg, all_item_agg, transpose_b=True)
        return scores

    # =========================================================
    # TRAIN API
    # =========================================================
    def fit_local_epoch(self, loader, local_id):
        n_batch = max(1, int(loader.n_C[local_id]) // self.batch_size + 1)
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
                    self.users: np.asarray(users, dtype=np.int32),
                    self.pos_items: np.asarray(pos_items, dtype=np.int32),
                    self.neg_items: np.asarray(neg_items, dtype=np.int32),
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
        n_batch = max(1, int(loader.n_train) // self.batch_size + 1)
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
                    self.users: np.asarray(users, dtype=np.int32),
                    self.pos_items: np.asarray(pos_items, dtype=np.int32),
                    self.neg_items: np.asarray(neg_items, dtype=np.int32),
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
            variables = tf.global_variables()
            values = self.sess.run(variables)
            names = [v.name for v in variables]
        return {name: value for name, value in zip(names, values)}

    def set_state(self, state):
        with self.graph.as_default():
            assigns = []
            for var in tf.global_variables():
                if var.name in state:
                    assigns.append(tf.assign(var, state[var.name]))
            if assigns:
                self.sess.run(assigns)

    def export_local_embeddings(self):
        """
        Return local shard embeddings for cache/checkpoint usage.
        """
        user_vals = self.sess.run(self.weights["user_embedding_list"])
        item_vals = self.sess.run(self.weights["item_embedding_list"])
        return {
            "user_embedding_list": user_vals,
            "item_embedding_list": item_vals,
        }

    def load_local_embeddings(self, state):
        """
        Load only local shard embeddings, useful after retraining affected shards.
        """
        with self.graph.as_default():
            assigns = []
            if "user_embedding_list" in state:
                for var, val in zip(self.weights["user_embedding_list"], state["user_embedding_list"]):
                    assigns.append(tf.assign(var, val))
            if "item_embedding_list" in state:
                for var, val in zip(self.weights["item_embedding_list"], state["item_embedding_list"]):
                    assigns.append(tf.assign(var, val))
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

    def export_local_embeddings(self):
        return self.model.export_local_embeddings()

    def load_local_embeddings(self, state):
        self.model.load_local_embeddings(state)

    def clone_fresh(self):
        return RecEraserBPRWrapper(self.cfg, self.n_users, self.n_items)

    def close(self):
        self.model.close()
