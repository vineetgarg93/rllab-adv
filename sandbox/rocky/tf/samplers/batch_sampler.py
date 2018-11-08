from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.initialize_all_variables())


class BatchSampler(BaseSampler):
    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.algo.env, self.algo.pro_policy, scope=self.algo.scope, adv_policy=self.algo.adv_policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_pro_params = self.algo.pro_policy.get_param_values()
        cur_adv_params = self.algo.adv_policy.get_param_values()

        if hasattr(self.algo.env,"get_param_values"):
            cur_env_params = self.algo.env.get_param_values()
        else:
            cur_env_params = None
        
        paths = parallel_sampler.sample_paths(
            pro_policy_params=cur_pro_params,
            env_params=cur_env_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
            adv_policy_params=cur_adv_params
        )

        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated
