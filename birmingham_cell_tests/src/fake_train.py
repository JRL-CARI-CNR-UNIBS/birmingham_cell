#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import rospkg
import os


class MyCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model"):
        super(MyCheckpointCallback, self).__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            for filename in os.listdir(self.save_path):
                if filename.startswith(self.name_prefix):
                    os.remove(os.path.join(self.save_path, filename))
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True

# from gymnasium import NormalizeObservation
# from stable_baselines3.common.vec_env import VecNormalize


test_name = "fake_1_"
epoch_number = 500
max_epoch_steps = 60
learning_start_steps = 40
train_freq = 1
learning_rate = 0.001
gamma = 0.9
total_timesteps = max_epoch_steps * epoch_number
model_save_freq = 1

rospack = rospkg.RosPack()
path = rospack.get_path('birmingham_cell_tests')

model_repo_path = path + '/model'
data_repo_path = path + '/data'
log_repo_path = path + '/log'
models_name = test_name + '_models_' + str(gamma) + '_' + str(learning_rate)
model_name = test_name + '_model_' + str(gamma) + '_' + str(learning_rate)
data_name = test_name + '_data_' + str(gamma) + '_' + str(learning_rate)
log_name = test_name + '_log_' + str(gamma) + '_' + str(learning_rate)
model_path = model_repo_path + '/' + model_name
data_path = data_repo_path + '/' + data_name
log_path = log_repo_path + '/' + log_name
models_repo_path = model_repo_path + '/' + models_name

env = gym.make('FakeEnv-v0',
               action_type='increment_value', 
               max_episode_steps=max_epoch_steps, 
               data_file_name=data_name,
            #    debug_mode=True)
               debug_mode=False)

# env = gym.NormalizeObservation(env)

#checkpoint_callback = CheckpointCallback(
#    save_freq=max_epoch_steps,
#    save_path=models_repo_path + '/',
#    name_prefix=model_name,
#    keep_only_best=False
#)
checkpoint_callback = MyCheckpointCallback(save_freq=model_save_freq,
                                           save_path=models_repo_path + '/', 
                                           name_prefix=model_name)  

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", 
            env, 
            verbose=1, 
            # action_noise=action_noise,
            learning_rate=learning_rate,
            learning_starts=learning_start_steps, 
            tensorboard_log=log_path,
            train_freq=train_freq,
            gamma=gamma)

model.learn(total_timesteps=total_timesteps, log_interval=1, callback=checkpoint_callback)

model.save(model_path)
