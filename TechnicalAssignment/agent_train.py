import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import imageio
import os


class FrozenLake():
    """Custom class to train models"""

    def __init__(self, 
                 n_steps : int,
                 eval_freq: int,
                 model_dir: str,
                 n_eval_episodes: int) -> None:
        
        self.n_steps = n_steps
        self.eval_freq = eval_freq
        self.model_dir = model_dir
        self.n_eval_episodes = n_eval_episodes
        self.base_url = 'http://localhost:5005'
        # create dir to save models if not exists
        os.makedirs(self.model_dir, exist_ok=True)

    def get_env(self) -> None:
        env_id = 'FrozenLake-v1'
        map_name = '4x4'
        is_slippery = True
        render_mode = 'rgb_array'

        self.env = gym.make(env_id,
                            desc = None,
                            map_name = map_name,
                            is_slippery = is_slippery,
                            render_mode='rgb_array'
                            )
        self.env = Monitor(self.env)

    def train(self, 
              train_alg: type[BaseAlgorithm],
              model_path: str,
              **kwargs) -> BaseAlgorithm:
        
        eval_callback = EvalCallback(self.env,
                                     best_model_save_path = model_path,
                                     log_path = model_path,
                                     eval_freq=self.eval_freq,
                                     deterministic=True,
                                     render=False)
        # init model and policy
        model = train_alg('MlpPolicy',
                          self.env,
                          tensorboard_log=f'./{train_alg.__name__}_tensorboard/',
                          **kwargs,
                        #   verbose=1
                          )
        
        model.learn(total_timesteps=self.n_steps,
                    callback=eval_callback)
        
        model.save(os.path.join(model_path, 'trained_model'))

        return model
    
    def eval_no_api (self,
             model:BaseAlgorithm) -> tuple: # this method is used to evaluation without the FrozenLakeAPI
        
        mean_reward, std_reward = evaluate_policy(model, 
                                                  self.env,
                                                  n_eval_episodes=self.n_eval_episodes)
        
        return mean_reward, std_reward

    def load_best_model(self, 
                        train_alg: type[BaseAlgorithm], 

                        model_path: str) -> tuple:
        
        model = train_alg.load(os.path.join(model_path, 'best_model.zip'))

        return model

    def render_gif(self, 
                   model: BaseAlgorithm, 
                   gif_path: str, 
                   max_steps: int) -> None:
        
        images = []
        obs, info = self.env.reset()
        img = self.env.render()
        for _ in range(max_steps):
            images.append(img)
            action, _ = model.predict(obs)
            action = int(action) 
            obs, reward, done, truncated, info = self.env.step(action)
            img = self.env.render()
            if done or truncated:
                break
        imageio.mimsave(gif_path, [np.array(img) for img in images], fps=29)
       
if __name__ == '__main__':
    # train parameters
    n_steps = 1e5
    eval_freq = 1000
    model_dir = 'models/'
    n_eval_episodes = 100  

    #DQN hyperparams
    dqn_hyperparams =  {
        'learning_rate': 1e-4,
        'buffer_size': 1000000,
        'learning_starts': 1000,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'device': 'cuda'
    }

    # PPO hyperparameters
    ppo_hyperparams = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'device':'cuda'
    }


    # Init class obj
    agent = FrozenLake(n_steps, 
                       eval_freq, 
                       model_dir, 
                       n_eval_episodes
                       )
    
    
    # Create env
    agent.get_env()

    print('****** MODEL TRAININGS ******\n\n')
    # Train DQN
    dqn_model_path = os.path.join(model_dir, 'DQN')
    os.makedirs(dqn_model_path, exist_ok=True)
    dqn_model= agent.train(DQN, dqn_model_path, **dqn_hyperparams)
    dqn_model_best = agent.load_best_model(DQN, dqn_model_path)
    print("---> Training DQN '\u2713\n")

    # Train PPO
    ppo_model_path = os.path.join(model_dir, 'PPO')
    os.makedirs(ppo_model_path, exist_ok=True)
    ppo_model = agent.train(PPO, ppo_model_path, **ppo_hyperparams)
    ppo_model_best = agent.load_best_model(PPO, ppo_model_path)
    print("---> Training PPO \u2713\n")


    print("****** MODEL EVALUATION ****** \n\n") # evaluation without the FrozenLakeAPI

    mean_reward_dqn, std_reward_dqn = agent.eval_no_api(dqn_model_best)
    print(f'DQN - Mean reward: {mean_reward_dqn} +/- {std_reward_dqn}')
    mean_reward_ppo, std_reward_ppo = agent.eval_no_api(ppo_model_best)
    print(f'PPO - Mean reward: {mean_reward_ppo} +/- {std_reward_ppo}')


    print('***** CREATE GIFS ******')

    agent.render_gif(dqn_model_best, "dqn_agent.gif", max_steps=350)
    agent.render_gif(ppo_model_best, "ppo_agent.gif", max_steps=350)







