from abc import ABC, abstractmethod
from RL.environment.Observ import Observ
from RL.environment.Action import simpleAction


class config:
    __slots__ = ['nn_policy', 'nn_quality', 'centralized']

    def __init__(self,
                 nn_policy,
                 nn_quality,
                 centralized=True,
                 ):
        self.nn_policy = nn_policy
        self.nn_quality = nn_quality
        self.centralized = centralized

class abcAgent(ABC):
    def __init__(self,
                 c_config,
                 observ_scheme,
                 action_scheme
                 ):
        self.config = c_config
        self.observ_scheme = observ_scheme
        self.action_scheme = action_scheme

    @abstractmethod
    def take_action(self):
        return NotImplementedError

    @abstractmethod
    def calculate_loss(self):
        return NotImplementedError
