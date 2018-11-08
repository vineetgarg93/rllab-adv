import pickle

import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy

        pro_policy = self.algo.pro_policy
        adv_policy = self.algo.adv_policy

        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            pro_policy.reset(dones)
            adv_policy.reset(dones)
            pro_actions, pro_agent_infos = pro_policy.get_actions(obses)
            adv_actions, adv_agent_infos = adv_policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(pro_actions, adv_actions)
            env_time += time.time() - t

            t = time.time()

            pro_agent_infos = tensor_utils.split_tensor_dict_list(pro_agent_infos)
            adv_agent_infos = tensor_utils.split_tensor_dict_list(adv_agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if pro_agent_infos is None:
                pro_agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if adv_agent_infos is None:
                adv_agent_infos = [dict() for _ in range(self.vec_env.num_envs)]

            for idx, observation, action, reward, env_info, pro_agent_info, adv_agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, pro_agent_infos, adv_agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        pro_actions=[],
                        adv_actions=[],
                        rewards=[],
                        env_infos=[],
                        pro_agent_infos=[],
                        adv_agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["pro_actions"].append(pro_action)
                running_paths[idx]["adv_actions"].append(adv_action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["pro_agent_infos"].append(pro_agent_info)
                running_paths[idx]["adv_agent_infos"].append(adv_agent_info)
                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        pro_actions=self.env_spec.action_space.flatten_n(running_paths[idx]["pro_actions"]),
                        adv_actions=self.env_spec.action_space.flatten_n(running_paths[idx]["adv_actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        pro_agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["pro_agent_infos"]),
                        adv_agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["adv_agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths
