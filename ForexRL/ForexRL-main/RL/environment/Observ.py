import numpy
import gym


class Observ:
    def __init__(self,
                 data_observ: numpy.ndarray,
                 window_size: int,
                 number_agents: int,
                 similar_observ: bool = True):
        self.data_observ = data_observ
        self.similar_observ = similar_observ
        self.number_agents = number_agents
        self.window_size = window_size

    def provide(self,
                index_current_tick: int,
                ):
        observes = None
        if self.similar_observ:
            observes = self.data_observ[(index_current_tick - self.window_size):index_current_tick, ...]
        else:
            for i in range(self.number_agents):
                observes.append(self.data_observ[(index_current_tick - self.window_size):index_current_tick, i, ...])

        return observes

    @property
    def obs_space(self):
        obs_space = []
        if self.similar_observ:
            obs_shape = (self.window_size,) + self.data_observ.shape[1:]
            #print(obs_shape)
            return gym.spaces.Box(
                low=-1 * numpy.inf,
                high=numpy.inf,
                shape=obs_shape,
                dtype=numpy.float32)
        else:
            for i in range(self.number_agents):
                obs_shape = (self.window_size,) + self.data_observ.shape[2:]
                #print(obs_shape)

                obs_space.append(gym.spaces.Box(
                    low=-1 * numpy.inf,
                    high=numpy.inf,
                    shape=obs_shape,
                    dtype=numpy.float32))

        return obs_space

# obs = numpy.random.rand(100, 5, 10)
#
# ob = Observ(obs, 10,5, similar_observ=True)
# print(ob.provide(40))
#
