import gym
import numpy as np
import torch
from Agent import Agent
from visualize import plotting
import matplotlib.pyplot as plt
import time

env_name = "LunarLanderContinuous-v2"
env = gym.make(env_name)

log_file_name = 'log_' + env_name + '.txt'

Test_mode = True
test_episode_number = 100
success_threshold = 220

seed = 2020
torch.manual_seed(seed)
env.seed(seed)
np.random.seed(seed)

Max_Episode = 2000
batch_size = 32

agent = Agent(gamma=0.99, tau=0.001, actorlr=3e-4, criticlr=1e-3, variance=0.1, action_dim=env.action_space.sample().shape[0],
              mem_size=1000000, batch_size=batch_size, pos_dim=env.observation_space.shape[0])

rew_cum = []
rew_per_episode = []
max_average = 0

rewards = np.zeros(100)
tot_step = 0

if Test_mode:
    file_test = open("Test_model.txt", 'w')
    rew_test = []
    agent.critic.load_state_dict(torch.load('./a/critic_network_611.ckpt'))
    agent.actor.load_state_dict(torch.load('./a/actor_network_611.ckpt'))
    agent.variance = 0.0
    success_number = 0
    for i in range(test_episode_number):
        state = env.reset()
        tot_rew = 0
        while True:
            env.render()
            action = agent.action_selection(state)
            new_state, reward, done, _ = env.step(action)
            tot_rew += reward
            state = new_state
            if done:
                result = 1 * (tot_rew > success_threshold)
                success_number += result
                file_test.writelines("Episode: {} Reward: {} Success: {}".format(i, tot_rew, result))
                print("Episode: {} Reward: {} Success: {}".format(i, tot_rew, result))
                rew_test.append(tot_rew)
                break
    plt.plot(rew_test)
    plt.title("Test Mode Reward")
    plt.xlabel("epoch")
    print("Success Rate: ", str(success_number / test_episode_number))
else:
    file = open(log_file_name, "w")

    for episode in range(Max_Episode):

        start = time.time()
        state = env.reset()
        episode_reward = 0

        while True:
            tot_step += 1
            action = agent.action_selection(state)
            new_state, reward, done, _ = env.step(action)
            agent.memory.store(state=state, action=action, reward=reward, new_state=new_state, terminal=done)

            if tot_step > batch_size:
                agent.update()
            state = new_state
            episode_reward += reward

            if done:
                end = time.time()
                rew_per_episode.append(episode_reward)
                rewards = np.concatenate((rewards, np.expand_dims(episode_reward, 0)))
                print("episode: {} reward: {} average reward: {} elapsed time: {}".format(episode, np.round(episode_reward, decimals=2),
                                                                                          np.sum(np.array(rew_per_episode)) / (episode + 1), end - start))
                file.writelines("episode: {} reward: {} average reward: {} elapsed time: {}".format(episode, np.round(episode_reward, decimals=2),
                                                                                                    np.sum(np.array(rew_per_episode)) / (episode + 1),
                                                                                                    end - start))
                file.write("\n")
                break

        if np.mean(np.array(rew_per_episode)[-30:]) > 220 and np.mean(np.array(rew_per_episode)[-30:]) > max_average:
            max_average = np.mean(np.array(rew_per_episode)[-30:])
            print("New average: ", max_average)
            print("Saving in episode: ", episode)
            torch.save(agent.actor.state_dict(), 'actor_network_'+str(episode)+'.ckpt')
            torch.save(agent.critic.state_dict(), 'critic_network_'+str(episode)+'.ckpt')

    plotting(file_name=log_file_name, env_name=env_name)
