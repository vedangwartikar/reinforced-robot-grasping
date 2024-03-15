import json
import numpy as np
import csv
from gym import spaces
from datetime import datetime
from stable_baselines import DQN, DDPG

from env.kuka_diverse_gym_env import KukaDiverseObjectEnv as SpecificShapeKukaDiverseObjectEnv


DATE = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
CSV_FILE = '../metrics.csv'

class Test:
    def __init__(self):
        self.config = self.load_config()
        self.log_dir = f"./models/{self.config['algorithm']}_{DATE}/"
        self.csv_file_name = CSV_FILE

    # Load config variables for training the model
    def load_config(self, config_path="../config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    # Setup Gym environment before fitting the algorithm
    def setup_environment(self):
        DISCRETE = True if self.config['algorithm'] == "DQN" else False
        env = SpecificShapeKukaDiverseObjectEnv(renders=True, isDiscrete=DISCRETE)
        return env
    
    # Load the model for testing the success ratio
    def setup_model(self, env):
        if self.config['algorithm'] == "DQN":
            model = DQN.load(self.config['model_dir'], env=env)


        if self.config['algorithm'] == "DDPG":
            env.action_space = spaces.Box(low=-1, high=1, shape=(4,))

            model = DDPG.load(self.config['model_dir'], env=env)
        
        return model

    # Calculate the success ratio over n test episodes
    def calculate_success_ratio(self, model, num_episodes):
        env = model.get_env()
        successratio = 0
        for _ in range(num_episodes):
            done = False
            obs = env.reset()
            while not done:
                action, states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if reward == 1:
                    print("Target object picked up...")
                    successratio += 1

        successratio = successratio / num_episodes

        header = ["Algorithm", "Timesteps", "Num episodes", "Success ratio"]

        with open(self.csv_file_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            if csv_file.tell() == 0:
                csv_writer.writerow(header)

            csv_writer.writerow([self.config['algorithm'], self.config['timesteps'], num_episodes, successratio])

        return successratio

# Load the config, setup Gym env, compute the success ratios for test episodes
if __name__ == "__main__":
    test = Test()
    config = test.load_config()
    env = test.setup_environment()
    model = test.setup_model(env)
    num_episodes = config['test_episodes']
    successratio = test.calculate_success_ratio(model, num_episodes)
    print("Algorithm: ", config['algorithm'])
    print("Number of episodes: ", num_episodes)
    print("Success rate: ", successratio)