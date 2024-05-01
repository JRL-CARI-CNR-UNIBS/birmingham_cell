#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import birmingham_envs
import sys

from stable_baselines3 import TD3, SAC, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

import rospkg
import os
import yaml
import datetime

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


if __name__ == '__main__':

    params_path = sys.argv[1]

    possible_model_type = ['td3','sac','ddpg']

    rospack = rospkg.RosPack()
    path = rospack.get_path('birmingham_cell_tests')
    file_path = path + '/' + params_path

    with open(file_path) as file:
        params = yaml.safe_load(file)

    if 'model_types' in params:
        for model_type in params['model_types']:
            if not model_type in possible_model_type:
                print('Model_type not in the possible model list.')
                exit(0)
    else:
        print('Model_types is empty')
        exit(0)

    if 'name_space' in params:
        name_space = params['name_space']
    else:
        name_space = 'generic_tests'
    if 'verbose' in params:
        verbose = params['verbose']
    else:
        verbose = 0
             
    if 'distance_threshold' in params:
        distance_threshold = params['distance_threshold']
    else:
        distance_threshold = 0.02
    if 'force_threshold' in params:
        force_threshold = params['force_threshold']
    else:
        force_threshold = 100
    if 'debug_mode' in params:
        debug_mode = params['debug_mode']
    else:
        debug_mode = False
    if 'step_print' in params:
        step_print = params['step_print']
    else:
        step_print = False
    if 'only_pos_success' in params:
        only_pos_success = params['only_pos_success']
    else:
        only_pos_success = True

    data = datetime.datetime.now()
    test_name = name_space + 'tests'
    
    log_repo_path = path + '/log' 
    models_repo_path = path + '/model/'

    test_number = 0
    total_test = len(params['max_epoch_steps']) * len(params['learning_rate']) * len(params['gamma'])
    print('Total tests: ' + str(total_test))
    for env_type in params['env_type']:
        for max_epoch_steps in params['max_epoch_steps']:
            for learning_rate in params['learning_rate']:
                for gamma in params['gamma']:
                    for model_type in params['model_types']:
                        test_number += 1
                        print('Test ' + str(test_number))
                        # model_name = test_name + '_' + str(max_epoch_steps) + '_' + str(learning_rate) + '_' + str(gamma)
                        # log_name = test_name + '_' + str(max_epoch_steps) + '_' + str(learning_rate) + '_' + str(gamma)
                        if 'name_space' in params:
                            name_space = params['name_space'] + '/'
                        else:
                            name_space = ''
                        model_name = name_space + env_type + '/'  + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma)
                        log_name = name_space + env_type + '/' + str(max_epoch_steps) + '/' + str(learning_rate) + '/' + str(gamma)
                        model_path = models_repo_path + '/' + model_name
                        log_path = log_repo_path + '/' + log_name
                        if env_type == 'fake':
                            env = gym.make('FakeEnv-v0',
                                        action_type='increment_value',
                                        epoch_len = max_epoch_steps,
                                        max_episode_steps=max_epoch_steps)
                        elif env_type == 'easy':
                            env = gym.make('EasyEnv-v0',
                                        action_type='increment_value', 
                                        env_dimension=1,
                                        max_episode_steps=max_epoch_steps)
                        elif env_type == 'connection':
                            env = gym.make('ConnectionEnv-v0', 
                                            action_type='increment_value', 
                                            distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake':
                            print('In realistic_fake')
                            env = gym.make('RealisticFakeEnv-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_and_pos':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param_and_pos',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_pos_and_reward':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param_pos_and_reward',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'realistic_fake_param_pos_and_reward_history':
                            print('In ' + env_type)
                            env = gym.make('RealisticFakeEnv2-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            obs_type = 'param_pos_and_reward_history',
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'random_real_fake':
                            print('In random_real_fake')
                            env = gym.make('RandomRealFakeEnv-v0', 
                                            action_type='increment_value', 
                                            # distance_threshold=distance_threshold,
                                            force_threshold=force_threshold,
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        elif env_type == 'generic_real_fake':
                            print('In generic_real_fake')
                            env = gym.make('GenericRealFakeEnv-v0', 
                                            action_type='increment_value', 
                                            debug_mode=debug_mode,
                                            step_print=step_print,
                                            only_pos_success=only_pos_success,
                                            epoch_len = max_epoch_steps,
                                            max_episode_steps=max_epoch_steps)
                        else:
                            print('Env_type not in the possible env list.')
                            exit(0)  

                        n_actions = env.action_space.shape[-1]
                        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                        if (model_type == 'td3'):
                            model = TD3("MlpPolicy", 
                                        env, 
                                        verbose=verbose,
                                        action_noise=action_noise,
                                        learning_rate=learning_rate,
                                        tensorboard_log=log_path,
                                        gamma=gamma,
                                        )
                        if (model_type == 'sac'):
                            model = SAC("MlpPolicy", 
                                        env, 
                                        verbose=verbose,
                                        action_noise=action_noise,
                                        learning_rate=learning_rate,
                                        tensorboard_log=log_path,
                                        gamma=gamma,
                                        )
                        if (model_type == 'ddpg'):
                            model = DDPG("MlpPolicy", 
                                        env, 
                                        verbose=verbose,
                                        action_noise=action_noise,
                                        learning_rate=learning_rate,
                                        tensorboard_log=log_path,
                                        gamma=gamma,
                                        )
                            
                        if 'model_save_freq' in params:
                            # checkpoint_callback = MyCheckpointCallback(save_freq=params['model_save_freq'],
                            #                         save_path=model_path + '/', 
                            #                         name_prefix=name_space)
                            checkpoint_callback = CheckpointCallback(
                                save_freq=params['model_save_freq'],
                                save_path=model_path + '/',
                                name_prefix=name_space,
                            )
                            model.learn(total_timesteps=params['total_timesteps'], 
                                        log_interval=1, 
                                        callback=checkpoint_callback,
                                        )
                            model.save(model_path)
                            
                        else:
                            model.learn(total_timesteps=params['total_timesteps'], 
                                    log_interval=1, 
                                    )
                           
