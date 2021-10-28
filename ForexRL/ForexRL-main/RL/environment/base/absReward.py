from abc import ABC, abstractmethod


class absReward(ABC):
    '''
    abstract method Reward
    '''



    @abstractmethod
    def calculate_reward(self, *step_args):
        """

        :param step_args: index_current_step,
                          action_current_step,
                          action_history,
                          ...

        :return: list of reward for each agent


        """
        return NotImplementedError


# class configReward:
#     __slots__ = ['unit_reward', 'type_reward', 'same_reward']
#
#     def __init__(self,
#                  unit_reward: int = 1,
#                  type_reward: str = 'discrete',
#                  same_reward: bool = True,
#                  ):
#         self.unit_reward = unit_reward
#         self.type_reward = type_reward
#         self.same_reward = same_reward
#     pass



