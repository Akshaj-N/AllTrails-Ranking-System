# Must do pip install 'gymnasium[atari,accept-rom-license]'
import gym
import gym_backgammon

print(gym.envs.registry.keys())


# Training loop

env = gym.make("Backgammon-v0")




env.close()


