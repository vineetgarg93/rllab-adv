

import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs

        self._pro_action_space = envs[0].pro_action_space
        self._adv_action_space = envs[0].adv_action_space

        # self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_pro, action_adv):

        class temp_action(object): pro=None; adv=None;

        all_results = []
        cum_a = temp_action()

        for (a1, a2, env) in zip(action_pro, action_adv, self.envs):
            cum_a.pro = a1
            cum_a.adv = a2
            all_results.append(env.step(cum_a))
        # all_results = [env.step(a) for (a1, a2, env) in zip(action_pro, action_adv, self.envs)]

        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    # @property
    # def action_space(self):
    #     return self._action_space

    @property
    def pro_action_space(self):
        return self._pro_action_space

    @property
    def adv_action_space(self):
        return self._adv_action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
