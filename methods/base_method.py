from abc import ABC, abstractmethod


class BaseUnlearningMethod(ABC):
    def __init__(self, cfg, loader, model):
        self.cfg = cfg
        self.loader = loader
        self.model = model

    @abstractmethod
    def initial_train(self):
        raise NotImplementedError

    @abstractmethod
    def unlearn(self, users_to_remove):
        raise NotImplementedError
