import os
import random
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class RecEraserLightGCNTF:
    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = int(n_users)
        self.n_items = int(n_items)

        self.model_type = "RecEraser_LightGCN"
        self.lr = float(getattr(cfg, "lr", 1e-3))
        self.emb_dim = int(getattr(cfg, "emb_dim", 64))
        self.attention_size = max(1, self.emb_dim // 2)
        self.batch_size = int(getattr(cfg, "batch_size", 256))
        self.decay = float(getattr(cfg, "reg_lambda", 1e-4))
        self.verbose = bool(getattr(cfg, "print_loss", True))

        self.num_local = int(getattr(cfg, "shard_num", 3))
        self.n_fold = int(getattr(cfg, "n_fold", 3))
        self.n_layers = int(getattr(cfg, "gcn_layers", 2))
        self.keep_prob = float(getattr(cfg, "dropout", 1.0))
        self.node_dropout_flag = False
        self.seed = int(getattr(cfg, "seed", 2024))

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.norm_adj = None
        self.local_norm_adj = {}
        self.global_norm_adj = None
        self.local_adj_tuples = {}
        self.global_adj_tuple = None
        self.local_feed_templates = {}
        self.agg_feed_template = None
        self.predict_feed_template = None

        # Keep signatures to avoid stale cache after unlearning
        self._global_adj_signature = None
        self._local_adj_signatures = {}

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.users = tf.placeholder(tf.int32, shape=(None,), name="users")
            self.pos_items = tf.placeholder(tf.int32, shape=(None,), name="pos_items")
            self.neg_items = tf.placeholder(tf.int32, shape=(None,), name="neg_items")

            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.node_dropout = tf.placeholder_with_default([0.0], shape=[None], name="node_dropout")
            self.mess_dropout = tf.placeholder_with_default([0.0], shape=[None], name="mess_dropout")

            self.weights = self._init_weights()

            self.local_adj_placeholders = [
                tf.sparse_placeholder(tf.float32, name=f"local_adj_{i}")
                for i in range(self.num_local)
            ]
            self.global_adj_placeholder = tf.sparse_placeholder(tf.float32, name="global_adj")

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
                self.batch_ratings,
                self.u_w,
            ) = self.train_agg_model2()

            self.init_op = tf.global_variables_initializer()

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True

        self.sess = tf.Session(graph=self.graph, config=session_config)
        self.sess.run(self.init_op)

    # =========================================================
    # GRAPH BUILD
    # =========================================================
    def _init_weights(self):
        all_weights = {}
        initializer = tf.random_normal_initializer(stddev=0.01)

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

            fold = tf.sparse_slice(adj_sp_tensor, [start, 0], [end - start, self.n_users + self.n_items])
            A_fold_hat.append(fold)

        return A_fold_hat

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        return mf_loss

    def _create_lightgcn_embed_local(self, local_num):
        A_fold_hat = self._split_A_hat_from_placeholder(self.local_adj_placeholders[local_num])

        ego_embeddings = tf.concat(
            [self.weights["user_embedding"][:, local_num], self.weights["item_embedding"][:, local_num]],
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

    def _create_lightgcn_embed(self, u_e, i_e):
        A_fold_hat = self._split_A_hat_from_placeholder(self.global_adj_placeholder)

        ego_embeddings = tf.concat([u_e, i_e], axis=0)
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

    def train_single_model(self, local_num):
        ua_embeddings, ia_embeddings = self._create_lightgcn_embed_local(local_num)

        u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.neg_items)

        u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["user_embedding"][:, local_num], self.users)
        pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"][:, local_num], self.pos_items)
        neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"][:, local_num], self.neg_items)

        mf_loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        regularizer = (
            tf.nn.l2_loss(u_g_embeddings_pre) +
            tf.nn.l2_loss(pos_i_g_embeddings_pre) +
            tf.nn.l2_loss(neg_i_g_embeddings_pre)
        )
        regularizer = regularizer / tf.cast(tf.maximum(tf.shape(self.users)[0], 1), tf.float32)
        emb_loss = self.decay * regularizer
        loss = mf_loss + emb_loss

        batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return opt, loss, mf_loss, emb_loss, batch_ratings

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

    def train_agg_model2(self):
        user_local_embs = tf.stop_gradient(self.weights["user_embedding"])
        item_local_embs = tf.stop_gradient(self.weights["item_embedding"])

        u_es = tf.einsum("abc,bcd->abd", user_local_embs, self.weights["trans_W"]) + self.weights["trans_B"]
        i_es = tf.einsum("abc,bcd->abd", item_local_embs, self.weights["trans_W"]) + self.weights["trans_B"]

        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, _ = self.attention_based_agg(i_es, 1)

        ua_embeddings, ia_embeddings = self._create_lightgcn_embed(u_e, i_e)

        u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.neg_items)

        u_drop = tf.nn.dropout(u_g_embeddings, keep_prob=self.dropout_keep_prob)
        mf_loss = self.create_bpr_loss(u_drop, pos_i_g_embeddings, neg_i_g_embeddings)
        loss = mf_loss

        batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return opt, loss, mf_loss, batch_ratings, u_w

    # =========================================================
    # ADJ BUILD / CACHE
    # =========================================================
    def _matrix_signature(self, mat):
        mat = mat.tocsr()
        return (mat.shape[0], mat.shape[1], int(mat.nnz))

    def _reset_adj_cache(self):
        self.norm_adj = None
        self.local_norm_adj = {}
        self.global_norm_adj = None
        self.local_adj_tuples = {}
        self.global_adj_tuple = None
        self.local_feed_templates = {}
        self.agg_feed_template = None
        self.predict_feed_template = None
        self._global_adj_signature = None
        self._local_adj_signatures = {}

    def _prepare_adj_cache(self, loader, force=False):
        if force:
            self._reset_adj_cache()

        if not (hasattr(loader, "get_adj_mat") and hasattr(loader, "get_adj_mat_local")):
            raise ValueError("Loader must provide get_adj_mat() and get_adj_mat_local() for RecEraserLightGCN.")

        global_adj = loader.get_adj_mat()
        local_adjs = [loader.get_adj_mat_local(lid) for lid in range(self.num_local)]

        new_global_sig = self._matrix_signature(global_adj)
        new_local_sigs = {lid: self._matrix_signature(local_adjs[lid]) for lid in range(self.num_local)}

        if (
            self.global_adj_tuple is not None
            and self._global_adj_signature == new_global_sig
            and self._local_adj_signatures == new_local_sigs
        ):
            return

        self.global_norm_adj = global_adj
        self.global_adj_tuple = self._convert_sp_mat_to_sp_tensor(self.global_norm_adj)

        self.local_norm_adj = {}
        self.local_adj_tuples = {}
        for lid in range(self.num_local):
            self.local_norm_adj[lid] = local_adjs[lid]
            self.local_adj_tuples[lid] = self._convert_sp_mat_to_sp_tensor(local_adjs[lid])

        self._global_adj_signature = new_global_sig
        self._local_adj_signatures = new_local_sigs

        self.local_feed_templates = {}
        eye_tuple = self._convert_sp_mat_to_sp_tensor(
            sp.eye(self.n_users + self.n_items, dtype=np.float32).tocsr()
        )

        for local_id in range(self.num_local):
            base_feed = {
                self.global_adj_placeholder: self.global_adj_tuple,
            }
            for lid in range(self.num_local):
                base_feed[self.local_adj_placeholders[lid]] = self.local_adj_tuples.get(lid, eye_tuple)
            self.local_feed_templates[local_id] = base_feed

        self.agg_feed_template = {
            self.global_adj_placeholder: self.global_adj_tuple,
        }
        for lid in range(self.num_local):
            self.agg_feed_template[self.local_adj_placeholders[lid]] = self.local_adj_tuples.get(lid, eye_tuple)

        self.predict_feed_template = dict(self.agg_feed_template)

    def _local_feed(self, loader, local_id, users, pos_items, neg_items):
        self._prepare_adj_cache(loader)
        feed = dict(self.local_feed_templates[local_id])
        feed.update({
            self.users: users,
            self.pos_items: pos_items,
            self.neg_items: neg_items,
            self.dropout_keep_prob: self.keep_prob,
        })
        return feed

    def _agg_feed(self, loader, users, pos_items, neg_items):
        self._prepare_adj_cache(loader)
        feed = dict(self.agg_feed_template)
        feed.update({
            self.users: users,
            self.pos_items: pos_items,
            self.neg_items: neg_items,
            self.dropout_keep_prob: self.keep_prob,
        })
        return feed

    # =========================================================
    # TRAIN / PREDICT / STATE
    # =========================================================
    def fit_local_epoch(self, loader, local_id):
        # force refresh to avoid stale local/global graph after unlearn
        self._prepare_adj_cache(loader, force=True)

        n_batch = max(1, loader.n_C[local_id] // self.batch_size + 1)
        loss = mf_loss = reg_loss = 0.0
        eff_batch = 0

        for _ in range(n_batch):
            users, pos_items, neg_items = loader.local_sample(local_id)
            if len(users) == 0:
                continue

            feed = self._local_feed(loader, local_id, users, pos_items, neg_items)

            _, batch_loss, batch_mf_loss, batch_reg_loss = self.sess.run(
                [
                    self.opt_local[local_id],
                    self.loss_local[local_id],
                    self.mf_loss_local[local_id],
                    self.reg_loss_local[local_id],
                ],
                feed_dict=feed,
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
            eff_batch += 1

        if eff_batch == 0:
            return 0.0, 0.0, 0.0

        return float(loss / eff_batch), float(mf_loss / eff_batch), float(reg_loss / eff_batch)

    def fit_agg_epoch(self, loader):
        # force refresh to avoid stale graph after unlearn
        self._prepare_adj_cache(loader, force=True)

        n_batch = max(1, loader.n_train // self.batch_size + 1)
        loss = mf_loss = 0.0
        eff_batch = 0

        for _ in range(n_batch):
            users, pos_items, neg_items = loader.sample()
            if len(users) == 0:
                continue

            feed = self._agg_feed(loader, users, pos_items, neg_items)

            _, batch_loss, batch_mf_loss = self.sess.run(
                [
                    self.opt_agg,
                    self.loss_agg,
                    self.mf_loss_agg,
                ],
                feed_dict=feed,
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            eff_batch += 1

        if eff_batch == 0:
            return 0.0, 0.0

        return float(loss / eff_batch), float(mf_loss / eff_batch)

    def predict(self, user_id, loader=None):
        if loader is not None:
            self._prepare_adj_cache(loader, force=True)
        elif self.predict_feed_template is None:
            raise ValueError("Adjacency is not prepared. Run training before predict(), or pass loader to predict().")

        with self.graph.as_default():
            user_arr = np.full(shape=(self.n_items,), fill_value=int(user_id), dtype=np.int32)
            item_arr = np.arange(self.n_items, dtype=np.int32)
            dummy_neg = np.zeros(self.n_items, dtype=np.int32)

            feed = dict(self.predict_feed_template)
            feed.update({
                self.users: user_arr,
                self.pos_items: item_arr,
                self.neg_items: dummy_neg,
                self.dropout_keep_prob: 1.0,
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


class RecEraserLightGCNWrapper:
    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = n_users
        self.n_items = n_items
        self.model = RecEraserLightGCNTF(cfg, n_users, n_items)
        self._last_loader = None

    def fit_local(self, loader, local_id, epochs=None):
        self._last_loader = loader

        if epochs is None:
            epochs = getattr(self.cfg, "local_epochs", getattr(self.cfg, "epochs", 1))

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
                    f"[REC LIGHTGCN LOCAL][shard={local_id}][Epoch {epoch+1}/{epochs}] "
                    f"loss={loss:.6f}, mf={mf_loss:.6f}, reg={reg_loss:.6f}"
                )

        return last_stats

    def fit_agg(self, loader, epochs=None):
        self._last_loader = loader

        if epochs is None:
            epochs = getattr(self.cfg, "epoch_agg", getattr(self.cfg, "agg_epochs", 1))

        last_stats = None
        for epoch in range(epochs):
            loss, mf_loss = self.model.fit_agg_epoch(loader)
            last_stats = {
                "loss": loss,
                "mf_loss": mf_loss,
                "epoch": epoch + 1,
            }

            if getattr(self.cfg, "print_loss", False):
                print(
                    f"[REC LIGHTGCN AGG][Epoch {epoch+1}/{epochs}] "
                    f"loss={loss:.6f}, mf={mf_loss:.6f}"
                )

        return last_stats

    def predict(self, user_id):
        return self.model.predict(user_id, loader=self._last_loader)

    def get_state(self):
        return self.model.get_state()

    def set_state(self, state):
        self.model.set_state(state)

    def clone_fresh(self):
        return RecEraserLightGCNWrapper(self.cfg, self.n_users, self.n_items)

    def close(self):
        self.model.close()