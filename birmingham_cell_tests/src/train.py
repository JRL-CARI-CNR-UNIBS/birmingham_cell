import argparse
import json
import os
import numpy as np
from envs import CuttingEnvV2
from utils.io import get_datetime_string
from utils.training import exponential_schedule
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


def make_env_options():
    tcp_site_name = "cutter_tip"
    controller_options = {
        "ee_site_name": tcp_site_name,
        "stiffness": [400.0, 400.0, 400.0, 400.0, 400.0, 400.0],
        "null_space_damping": 10.0,
        "auto_compute_damping": True,
        "lower_action_limit": [-10. for _ in range(6)],
        "upper_action_limit": [10. for _ in range(6)],
        "controlled_directions": [0, 1, 2],
        "controlled_joints": [
            "kuka_joint_1",
            "kuka_joint_2",
            "kuka_joint_3",
            "kuka_joint_4",
            "kuka_joint_5",
            "kuka_joint_6",
            "kuka_joint_7",
        ],
    }
    num_teeth = 50
    cutting_model_kwargs = {
        "tcp_site_name": tcp_site_name,
        "pitch_angle": 2 * np.pi / num_teeth,
        "helix_angle": 0.0,
        "radius": 0.025,
        "disc_height": 0.0005,
        "num_flutes": num_teeth,
        "num_discs": 1,
        "spindle_speed": 1000,
    }

    env_options = {
        "controller": "EnergyTankImpedanceController",
        "controller_kwargs": controller_options,
        "frame_skip": 10,
        "time_limit": 10.0,
        "time_step": 0.002,
        "random_model": True,
        "random_target": True,
        "terminal_reward": -50.0,
        "max_compute_task_attempts": -1,
        "trajectory_time_limit": 10.0,
        "wait_until_done": True,
        "cutting_model_kwargs": cutting_model_kwargs,
    }

    return env_options


if __name__ == "__main__":
    # Command line arguments
    log_dir = os.path.join(".", "logs")
    agent_save_dir = os.path.join(".", "agents")

    parser = argparse.ArgumentParser(
        description="Trains an agent using the cutting Gym environment."
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=f"agent_{get_datetime_string()}",
        help="the name of the agent to train",
    )
    parser.add_argument(
        "-N",
        "--num_episodes",
        type=int,
        default=10000,
        help="the number of training episodes",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=f'{os.path.join(agent_save_dir, "env_options")}',
        help="the path to the JSON environment config to use",
    )
    parser.add_argument(
        "--vectorise",
        action="store_true",
        help="train the agents with [num_envs] environments in parallel",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="if vectorised, train using this many environments in parallel",
    )
    args = parser.parse_args()

    # Directory configuration
    agent_name = args.name
    tensorboard_log_dir = os.path.join(log_dir, agent_name)
    agent_save_path = os.path.join(agent_save_dir, agent_name)
    env_save_path = agent_save_path + ".env"
    json_save_path = agent_save_path + ".json"

    # Load environment settings and setup environment
    config_file_path = args.config_file
    if os.path.exists(config_file_path):
        print(f"Loading JSON config file '{config_file_path}'")
        with open(config_file_path, "r") as json_file:
            env_options = json.load(json_file)
    else:
        print("No JSON config file was provided, using default config.")
        env_options = make_env_options()

    if args.vectorise:
        env = SubprocVecEnv(
            [(lambda: Monitor(CuttingEnvV2(**env_options))) for _ in range(args.num_envs)]
        )
    else:
        env = DummyVecEnv([lambda: Monitor(CuttingEnvV2(**env_options))])
    # Rolling average normalisation wrapper of observations and rewards
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path="./logs/",
    #     log_path="./logs/",
    #     eval_freq=500,
    #     deterministic=True,
    #     render=False,
    # )

    checkpoint_callback = CheckpointCallback(
        save_freq=250000,
        save_path="./logs/",
        name_prefix=agent_name,
    )

    # Training
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, learning_rate=exponential_schedule(0.0003, 0.125), batch_size=1024, gamma=0.99)
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, learning_rate=exponential_schedule(0.0003, 0.125), batch_size=1024, gamma=0.99)
    max_steps_per_episode = int(
        env_options["time_limit"]
        / (env_options["frame_skip"] * env_options["time_step"])
    )
    num_steps = int(args.num_episodes * max_steps_per_episode)
    model.learn(num_steps, callback=checkpoint_callback)
    model.save(agent_save_path)
    env.save(env_save_path)
    try:
        model.save_replay_buffer(agent_save_path + "_expbuffer")
    except AttributeError:
        print("Model type can't save experience buffer since it doesn't have one, skipping.")
    # Save JSON even if already loaded, to keep record of the options used to train the agent.
    with open(json_save_path, "w") as json_file:
        json_file.write(json.dumps(env_options))

    # Evaluate for single episode
    obs = env.reset()
    for i in range(max_steps_per_episode):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
