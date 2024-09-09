import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from ev2gym.models.ev2gym_env import EV2Gym
# Choose a default reward function and state function or create your own!!!
from ev2gym.rl_agent.reward import profit_maximization, SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"
env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=reward_function,
               state_function=state_function)
# Initialize the RL agent
model = DDPG("MlpPolicy", env)
# Train the agent
# model.learn(total_timesteps=1_000_000,
#             progress_bar=True)
# Evaluate the agent
env = model.get_env()
obs = env.reset()
stats = []
for eps in range(1000):
    for step in range(1000):
        for user in range(10):
            action, _states = model.predict(obs, deterministic=True)

        # action concatenate
        obs, reward, done, info = env.step(action)

            if done:
                stats.append(info)

    for user in range(10):
        # Sample Data
        pass
        # Train model
        # Send model to server
    # server.aggregate()
    # server.upload_model()

