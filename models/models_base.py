from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, cfg, n_users, n_items):
        self.cfg = cfg
        self.n_users = n_users
        self.n_items = n_items

    @abstractmethod
    def fit_one_epoch(self, train_user_dict):
        pass

    def fit(self, train_user_dict, epochs=None):
        if epochs is None:
            epochs = getattr(self.cfg, "epochs", 1)

        last_stats = None
        for _ in range(epochs):
            last_stats = self.fit_one_epoch(train_user_dict)
        return last_stats

    @abstractmethod
    def predict(self, user_id):
        pass

    @abstractmethod
    def clone_fresh(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    def state_dict(self):
        return self.get_state()

    def load_state_dict(self, state):
        self.set_state(state)

    def close(self):
        pass