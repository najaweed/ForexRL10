from abc import ABC, abstractmethod


class configAction:
    __slots__ = ['number_agents', 'number_actions', 'type_action', 'similar_action']

    def __init__(self,
                 number_agents: int = 12,
                 number_actions: int = 3,
                 type_action: str = 'discrete',
                 similar_action: bool = True,
                 ):
        self.number_agents = number_agents
        self.number_actions = number_actions
        self.type_action = type_action
        self.similar_action = similar_action


class absAction(ABC):

    def __inti__(self,
                 config: configAction,
                 ):
        self.config = config

    @property
    @abstractmethod
    def action_space(self):
        return NotImplementedError

    @property
    def num_agents(self):
        return self.config.number_agents

    @property
    def number_actions(self):
        return self.config.number_actions
