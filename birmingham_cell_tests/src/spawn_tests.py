#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from inspect import TPFLAGS_IS_ABSTRACT
from typing import Any, Dict, Optional, Tuple

# Definizione dell'ambiente personalizzato per l'ottimizzazione dei parametri
class HyperparameterOptimizationEnv(gym.Env):
    def __init__(self):
        super(HyperparameterOptimizationEnv, self).__init__()
        
        # Definizione degli spazi di azioni e stati
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Inizializzazione dei parametri
        self.params = np.random.rand(3)

    def step(self, action):
        # Aggiorna i parametri in base all'azione
        self.params = action
        
        # Valutazione della configurazione dei parametri
        reward = self.evaluate_params(self.params)
        
        # Definizione di uno stato fittizio (può essere più dettagliato in casi reali)
        state = self.params
        
        # Assumiamo che l'episodio termini dopo ogni step
        done = True
        truncated = False
        info = {"is_success": False}
        return state, reward, done, truncated, info
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.params = np.random.rand(3)
        info = {"is_success": False}
        return self.params, info

    def evaluate_params(self, params):
        # Funzione di valutazione (fittizia) dei parametri
        # In pratica, questa funzione valuterebbe le prestazioni del modello con i parametri dati
        return -np.sum((params - 0.5)**2)  # Simuliamo una funzione obiettivo

# Creazione dell'ambiente
env = HyperparameterOptimizationEnv()

# Creazione del modello PPO
model = PPO('MlpPolicy', env, verbose=1)

# Numero di timesteps per l'addestramento
total_timesteps = 1000

# Addestramento del modello
model.learn(total_timesteps=total_timesteps)

# Test del modello addestrato
obs, info = env.reset()
action, _states = model.predict(obs)
print("Migliori parametri trovati:", action)



exit(0)

import gymnasium as gym
import numpy as np
import birmingham_envs

from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import rospkg
import os

env = gym.make('ForceGraspEnv-v0', 
                epoch_len = 25,
                max_episode_steps=25)

env.reset()
# env = gym.make('GeneralEnv-v0', 
#                 epoch_len = 25,
#                 max_episode_steps=10000,
#                 space_dimension=6,
#                 history_len=10,
#                 single_threshold=0.01,
#                 use_reward=False,
#                 )


# env = gym.make('StaticConnectionEnv-v0',
#                action_type='target_value', 
#                max_episode_steps=25, 
#                obj_model_name='can',
#                tar_model_name='hole',
#             #    obj_model_height=0.1,
#             #    obj_model_length=0.04,
#             #    obj_model_width =0.04,
#             #    tar_model_height=0.06,
#             #    tar_model_length=0.045,
#             #    tar_model_width =0.045,
#                debug_mode=False)

# for i in range(10):
#     print(i+1)
#     env.reset()

exit(0)
action = [0.0,0.0,0.0,0.0,0.0,0.0]

os.system('roslaunch birmingham_cell_tests load_test_params.launch')

env.step(np.array(action))