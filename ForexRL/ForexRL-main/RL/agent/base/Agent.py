from RL.agent.base.abcAgent import abcAgent
from RL.environment.Observ import Observ
from RL.environment.Action import simpleAction


# https://github.com/ChenglongChen/pytorch-DRL
# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict
# https://github.com/iffiX/machin
class configAgent:
    __slots__ = ['nn_policy', 'nn_quality', 'centralized']

    def __init__(self,
                 nn_policy,
                 nn_quality,
                 centralized=True,
                 ):
        self.nn_policy = nn_policy
        self.nn_quality = nn_quality
        self.centralized = centralized


class Agent(abcAgent):
    def __init__(self, config, observ_scheme: Observ, action_scheme: simpleAction):
        super().__init__(config, observ_scheme, action_scheme)
        self.config = config
        self.action_scheme = action_scheme
        self.observ_scheme = observ_scheme
        self.number_agents = self.action_scheme.num_agents

    def take_action(self):
        pass

    def calculate_loss(self):
        pass


ag = Agent(configAgent, Observ, simpleAction)
print(ag.config.nnModel)
