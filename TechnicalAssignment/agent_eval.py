# import libaries 
import gymnasium as gym
import numpy as np
import requests
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces

API_BASE_URL = 'http://localhost:5005'

# class to build custom env 
class FrozenLakeAPIEnv(gym.Env):
    """Custom Environment for post reuquests to FrozenLakeAPI"""

    def __init__(self, api_base_url=API_BASE_URL):
        super().__init__()
        self.action_space = spaces.Discrete(4)  # 4 discrete actions in FrozenLake
        self.observation_space = spaces.Discrete(16)  # 16 possible states in a 4x4 FrozenLake
        self.api_base_url = api_base_url
        self.game_id = None

    def new_game(self):
        # Send new_game request to start a new game
        response = requests.post(f"{self.api_base_url}/new_game").json()
        self.game_id = response['game_id']

    def reset(self, seed=None): # seed not used but need for the evaluation_policy
        try:
            if self.game_id is None:
                self.new_game()  # If no game_id is set, start a new game

            # Send reset request to the API to reset the environment
            response = requests.post(f"{self.api_base_url}/reset", json={'game_id': self.game_id}).json()
            obs = response['observation']
            obs_state = int(obs[0])  # Convert int
            return obs_state, {}  
        
        except Exception as e:
            raise RuntimeError(f"Failed to reset environment: {e}")

    def step(self, action):
        action = int(action)  # converted to int
        response = requests.post(f"{self.api_base_url}/step", json={'game_id': self.game_id, 'action': action}).json()

        if 'observation' not in response:
            raise RuntimeError(f"Invalid response from API: {response}")
        
        obs = response['observation']
        reward = response['reward']
        done = response['done']
        truncated = response['truncated']
        info = response['info']
        return obs, reward, done, truncated, info  # Return observation, reward, done, an empty dictionary for truncated, and info

if __name__ == '__main__':

    model_dir = 'models/' # models dir
    n_eval_episodes = 100  # Number of episodes to evaluate

    # Load the models
    dqn_model_best = DQN.load(f'{model_dir}/DQN/best_model')
    ppo_model_best = PPO.load(f'{model_dir}/PPO/best_model')

    # Create the custom FrozenLake API environment
    env = FrozenLakeAPIEnv()
    monitored_env = Monitor(env)
    print("****** MODEL EVALUATION ******\n")

    # Evaluate DQN
    mean_reward, std_reward = evaluate_policy(dqn_model_best, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    print(f'DQN - Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}')

    # Evaluate PPO
    mean_reward, std_reward = evaluate_policy(ppo_model_best, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    print(f'PPO - Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}')


## Resources

# custom env : https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
# tensorboard: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
# dqn : https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
# ppo: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# evaluation policy: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html



