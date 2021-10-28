import enum
import numpy as np
import gym

from RL.environment.base.absAction import absAction


class configAction:
    __slots__ = ['number_agents', 'number_actions', 'type_action', 'similar_action']

    def __init__(self,
                 number_agents: int = 12,
                 number_actions: int = 3,
                 type_action: str = 'continuous',
                 similar_action: bool = True,
                 ):
        self.number_agents = number_agents
        self.number_actions = number_actions
        self.type_action = type_action
        self.similar_action = similar_action


class simpleAction(absAction):

    def __init__(self,
                 config: configAction,
                 ):
        """
        Args:
            config: configuration of action space
        """
        absAction.__init__(self)
        self.config = config
        self.dict_position_action = {'HOLD': 0, 'BUY': 1, 'SELL': 2}

    @property
    def action_space(self):
        act_space = []
        if self.config.similar_action and self.config.type_action == 'discrete':
            act_space = gym.spaces.MultiDiscrete([self.config.number_actions \
                                                  for _ in range(self.config.number_agents)])
        elif self.config.similar_action and self.config.type_action == 'continuous':
            act_space = gym.spaces.Box(low=-1,
                                       high=1,
                                       shape=(self.config.number_agents,),
                                       dtype=np.float32)
        return act_space

    @staticmethod
    def get_position(action: int):
        if action == 0:
            return 'HOLD'
        elif action == 1:
            return 'BUY'
        elif action == 2:
            return 'SELL'
        else:
            pass

    @staticmethod
    def normalize_action(action: np.ndarray):

        normal_volumes = action/sum(abs(action))
        return np.round(normal_volumes,2)


# action_config1 = configAction(25)
#
# a = simpleAction(action_config1)
# act = a.action_space.sample().tolist()
# act = np.array(act)
# print(act)
# print(a.normalize_action(act))
# print(sum(abs(a.normalize_action(act))))