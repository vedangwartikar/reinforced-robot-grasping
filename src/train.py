import os
import json
from gym import spaces
from datetime import datetime
from stable_baselines import DQN, DDPG
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.deepq.policies import LnCnnPolicy
from stable_baselines.ddpg.policies import LnCnnPolicy as DDPG_LnCnnPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec

from env.kuka_diverse_gym_env import KukaDiverseObjectEnv as SpecificShapeKukaDiverseObjectEnv


DATE = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

class Train:
    def __init__(self):
        self.config = self.load_config()
        self.log_dir = f"../models/{self.config['algorithm']}_{DATE}/"

    # Load config variables for training the model
    def load_config(self, config_path="../config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    # Setup Gym environment before fitting the algorithm
    def setup_environment(self):
        set_global_seeds(self.config["random_seed"])
        DISCRETE = True if self.config['algorithm'] == "DQN" else False
        os.makedirs(self.log_dir, exist_ok=True)

        env = SpecificShapeKukaDiverseObjectEnv(renders=True, isDiscrete=DISCRETE)
        env = Monitor(env, os.path.join(self.log_dir, "monitor.csv"), allow_early_resets=True)

        return env

    # Pass the Gym environment to the algorithm - DQN / DDPG
    def setup_model(self, env):
        if self.config["algorithm"] == "DQN":
            model = DQN(
                LnCnnPolicy,
                env,
                tensorboard_log=f"{self.log_dir}/tensorboard_{self.config['algorithm']}_{DATE}/",
                gamma=self.config["discount"],
                learning_rate=self.config["lr"]
            )
        elif self.config["algorithm"] == "DDPG":
            env.action_space = spaces.Box(low=-1, high=1, shape=(4,))
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
            model = DDPG(DDPG_LnCnnPolicy, 
                env, 
                gamma=self.config["discount"], 
                actor_lr=self.config["lr"],
                critic_lr=self.config["lr"],
                param_noise=param_noise, 
                tensorboard_log=f"{self.log_dir}/tensorboard_{self.config['algorithm']}_{DATE}/" 
            )

        return model

# Access internal parameters which helps in saving the current best model
def get_model_params(model, **kwargs):
    if not hasattr(model, "_callback_vars"):
        model._callback_vars = dict(**kwargs)
    else:
        for name, val in kwargs.items():
            if name not in model._callback_vars:
                model._callback_vars[name] = val
    return model._callback_vars

# Save the best model by comparing the grasp success wrt. timesteps
def save_best_model(_locals, _globals):
    log_dir = f"../models/{config['algorithm']}_{DATE}/"
    model_params = get_model_params(
        _locals["self"], each_step=0, abs_total_reward=-float("inf")
    )

    if not model_params["each_step"] % 20:
        x, y = ts2xy(load_results(log_dir), "timesteps")
        # print(x[-20:], y[-20:])
        if len(x):
            abs_total_reward = sum(y[-100:])/100
            # print(abs_total_reward)
            if abs_total_reward > model_params["abs_total_reward"]:
                model_params["abs_total_reward"] = abs_total_reward
                print(f"Saving new best model at {x[-1]} timesteps...")
                model_save_path = os.path.join(log_dir, "best_model")
                _locals["self"].save(model_save_path)

    model_params["each_step"] += 1
    return True

# Load the config, create Gym env, train the RL algorithm and save the best model
if __name__ == "__main__":
    train = Train()
    config = train.load_config()
    env = train.setup_environment()
    model = train.setup_model(env)

    time_steps = config["timesteps"]
    model.learn(total_timesteps=time_steps, callback=save_best_model)