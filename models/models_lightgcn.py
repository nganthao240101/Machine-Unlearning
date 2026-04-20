import os
import random
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class LightGCN(object):
    """
    Base LightGCN model for normal training flows:
    - Full Retrain
    - SISA

    This is NOT the RecEraser-specific LightGCN model.
    """

    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = int(n_users)
        self.n_items = int(n_items)

        self.model_type = "LightGCN"
        self.lr = float(getattr(cfg, "lr", 1e-3))
        self.emb_dim = int(getattr(cfg, "emb_dim", 64))
        self.batch_size = int(getattr(cfg, "batch_size", 256))
        self.decay = float(getattr(cfg, "reg_lambda", 1e-4))
        self.verbose = bool(getattr(cfg, "print_loss", False))
        self.seed = int(getattr(cfg, "seed", 2024))

        self.n_fold = 10
        self.n_layers = int(getattr(cfg, "gcn_layers", 3))
        self.node_dropout_flag = False

        self.norm_adj = None
        self.norm_adj_tuple = None
        self.feed_template = None

        # cache key for current training graph
        self._cached_graph_signature = None

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.users = tf.placeholder(tf.int32, shape=(None,), name="users")
            self.pos_items = tf.placeholder(tf.int32, shape=(None,), name="pos_items")
            self.neg_items = tf.placeholder(tf.int32, shape=(None,), name="neg_items")

            self.node_dropout = tf.placeholder_with_default([0.0], shape=[None], name="node_dropout")
            self.mess_dropout = tf.placeholder_with_default([0.0], shape=[None], name="mess_dropout")

            self.norm_adj_placeholder = tf.sparse_placeholder(tf.float32, name="norm_adj")

            self.weights = self._init_weights()

            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

            self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
            self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
            self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

            self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["user_embedding"], self.users)
            self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"], self.pos_items)
            self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"], self.neg_items)

            self.batch_ratings = tf.matmul(
                self.u_g_embeddings,
                self.pos_i_g_embeddings,
                transpose_a=False,
                transpose_b=True
            )

            self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(
                self.u_g_embeddings,
                self.pos_i_g_embeddings,
                self.neg_i_g_embeddings
            )
            self.loss = self.mf_loss + self.emb_loss

            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.init_op = tf.global_variables_initializer()

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True
        self.sess = tf.Session(graph=self.graph, config=session_config)
        self.sess.run(self.init_op)

    def _init_weights(self):
        all_weights = {}
        initializer = tf.random_normal_initializer(stddev=0.01)

        all_weights["user_embedding"] = tf.Variable(
            initializer([self.n_users, self.emb_dim]),
            name="user_embedding"
        )
        all_weights["item_embedding"] = tf.Variable(
            initializer([self.n_items, self.emb_dim]),
            name="item_embedding"
        )
        return all_weights

    def _convert_sp_mat_to_sp_tensor(self, X):
        if not sp.isspmatrix_coo(X):
            X = X.tocoo()
        indices = np.vstack((X.row, X.col)).transpose().astype(np.int64)
        values = X.data.astype(np.float32)
        shape = X.shape
        return indices, values, shape

    def _split_A_hat_from_placeholder(self, adj_sp_tensor):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            fold = tf.sparse_slice(
                adj_sp_tensor,
                [start, 0],
                [end - start, self.n_users + self.n_items]
            )
            A_fold_hat.append(fold)

        return A_fold_hat

    def _create_lightgcn_embed(self):
        A_fold_hat = self._split_A_hat_from_placeholder(self.norm_adj_placeholder)

        ego_embeddings = tf.concat(
            [self.weights["user_embedding"], self.weights["item_embedding"]],
            axis=0
        )
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(self.u_g_embeddings_pre) +
            tf.nn.l2_loss(self.pos_i_g_embeddings_pre) +
            tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        )
        regularizer = regularizer / tf.cast(tf.shape(users)[0], tf.float32)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _normalize_adj_from_user_dict(self, user_dict):
        n_nodes = self.n_users + self.n_items
        rows = []
        cols = []
        data = []

        for u, items in user_dict.items():
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

    def _make_graph_signature(self, train_user_dict):
        """
        Create a lightweight deterministic signature for current train graph.
        Used to avoid rebuilding adjacency repeatedly when the train data
        has not changed across epochs.
        """
        n_users = len(train_user_dict)
        n_interactions = 0
        checksum = 0

        for u, items in train_user_dict.items():
            n_interactions += len(items)
            checksum ^= (hash((int(u), len(items))) & 0xffffffff)

        return (n_users, n_interactions, checksum)

    def _prepare_adj_cache(self, train_user_dict, force=False):
        signature = self._make_graph_signature(train_user_dict)

        if force:
            self.norm_adj = None
            self.norm_adj_tuple = None
            self.feed_template = None
            self._cached_graph_signature = None

        if self.feed_template is not None and self._cached_graph_signature == signature:
            return

        self.norm_adj = self._normalize_adj_from_user_dict(train_user_dict)
        self.norm_adj_tuple = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        self.feed_template = {
            self.norm_adj_placeholder: self.norm_adj_tuple
        }
        self._cached_graph_signature = signature

    def _sample_from_user_dict(self, user_dict):
        valid_users = [u for u, items in user_dict.items() if len(items) > 0]
        if len(valid_users) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )

        if self.batch_size <= len(valid_users):
            users = random.sample(valid_users, self.batch_size)
        else:
            users = [random.choice(valid_users) for _ in range(self.batch_size)]

        final_users, pos_items, neg_items = [], [], []
        for u in users:
            pos_pool = user_dict.get(u, [])
            if not pos_pool:
                continue

            pos_i = random.choice(pos_pool)

            neg_i = random.randint(0, self.n_items - 1)
            guard = 0
            while neg_i in pos_pool and guard < max(100, self.n_items * 2):
                neg_i = random.randint(0, self.n_items - 1)
                guard += 1

            if neg_i in pos_pool:
                continue

            final_users.append(int(u))
            pos_items.append(int(pos_i))
            neg_items.append(int(neg_i))

        return (
            np.asarray(final_users, dtype=np.int32),
            np.asarray(pos_items, dtype=np.int32),
            np.asarray(neg_items, dtype=np.int32),
        )

    def fit_one_epoch(self, train_user_dict, prepare_adj=False):
        """
        Train one epoch on the provided train_user_dict.

        prepare_adj=False is preferred when fit() already prepared adjacency once
        for the whole training stage.
        """
        if prepare_adj:
            self._prepare_adj_cache(train_user_dict, force=False)

        n_interactions = int(sum(len(v) for v in train_user_dict.values()))
        n_batch = n_interactions // self.batch_size + 1

        loss = mf_loss = emb_loss = reg_loss = 0.0
        eff_batch = 0

        for _ in range(n_batch):
            users, pos_items, neg_items = self._sample_from_user_dict(train_user_dict)
            if len(users) == 0:
                continue

            feed = dict(self.feed_template)
            feed.update({
                self.users: users,
                self.pos_items: pos_items,
                self.neg_items: neg_items,
            })

            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = self.sess.run(
                [self.opt, self.loss, self.mf_loss, self.emb_loss, self.reg_loss],
                feed_dict=feed
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += float(np.asarray(batch_reg_loss).reshape(-1)[0])
            eff_batch += 1

        if eff_batch == 0:
            return 0.0, 0.0, 0.0, 0.0

        return (
            float(loss / eff_batch),
            float(mf_loss / eff_batch),
            float(emb_loss / eff_batch),
            float(reg_loss / eff_batch),
        )

    def fit(self, train_user_dict, epochs=None):
        if epochs is None:
            epochs = int(getattr(self.cfg, "epochs", 1))

        epochs = max(1, int(epochs))

        # Prepare adjacency ONCE for this train stage / retrain stage
        self._prepare_adj_cache(train_user_dict, force=True)

        last_stats = None
        for epoch in range(epochs):
            loss, mf_loss, emb_loss, reg_loss = self.fit_one_epoch(
                train_user_dict,
                prepare_adj=False
            )
            last_stats = {
                "loss": loss,
                "mf_loss": mf_loss,
                "emb_loss": emb_loss,
                "reg_loss": reg_loss,
                "epoch": epoch + 1,
            }

            if self.verbose:
                print(
                    f"[LightGCN][Epoch {epoch+1}/{epochs}] "
                    f"loss={loss:.6f}, mf={mf_loss:.6f}, emb={emb_loss:.6f}, reg={reg_loss:.6f}"
                )
        return last_stats

    def predict(self, user_id):
        if self.feed_template is None:
            raise ValueError("Adjacency is not prepared. Run training before predict().")

        with self.graph.as_default():
            user_arr = np.full(shape=(self.n_items,), fill_value=int(user_id), dtype=np.int32)
            item_arr = np.arange(self.n_items, dtype=np.int32)
            dummy_neg = np.zeros(self.n_items, dtype=np.int32)

            feed = dict(self.feed_template)
            feed.update({
                self.users: user_arr,
                self.pos_items: item_arr,
                self.neg_items: dummy_neg,
            })

            scores = self.sess.run(self.batch_ratings, feed_dict=feed)
        return np.diag(scores)

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


class LightGCNWrapper:
    """
    Project wrapper used by:
    - FullRetrainMethod
    - SISAMethod
    """

    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = n_users
        self.n_items = n_items
        self.model = LightGCN(cfg, n_users, n_items)

    def fit(self, train_user_dict, epochs=None):
        return self.model.fit(train_user_dict, epochs=epochs)

    def fit_one_epoch(self, train_user_dict):
        return self.model.fit_one_epoch(train_user_dict, prepare_adj=True)

    def predict(self, user_id):
        return self.model.predict(user_id)

    def get_state(self):
        return self.model.get_state()

    def set_state(self, state):
        self.model.set_state(state)

    def clone_fresh(self):
        return LightGCNWrapper(self.cfg, self.n_users, self.n_items)

    def close(self):
        self.model.close()