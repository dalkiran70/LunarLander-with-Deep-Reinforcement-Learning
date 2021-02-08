from Agent import Agent
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

tau = 0.0001
lr = 2e-4
epsilon = 0.1
gamma = 0.99
SOLVED_THRESHOLD = 200
SUCCESS_THRESHOLD = 200
SUCCESS_INTERVAL = 30
env_name = "LunarLander-v2"
env = gym.make(env_name)
output_dim = env.action_space.n
input_dim = env.observation_space.sample().shape[0]
buffer_size = 1000000
batch_size = 32
filename = "Double_DQN_logs.txt"
file = open(filename, 'w')
train_mode = False
model_params_filename = "Solved_Q_Network.pt360"
TEST_EPISODE_NUMBER = 100

agent = Agent(epsilon=epsilon, input_dim=input_dim, hidden1=400, hidden2=300, output_dim=output_dim, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau)


if train_mode:
    episode_rewards = []
    cumulative_rewards = []
    solved = False
    max_rew = 0
    episode_number = 0

    while not solved:
        state = torch.tensor(env.reset())
        episode_rew = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, r, done, _ = env.step(action.item())
            episode_rew += r
            next_state = torch.tensor(next_state)
            r = torch.tensor(r)
            done = torch.tensor(done)
            agent.store(state, action, r, next_state, done)
            if episode_number > 5:
                agent.update()
            state = next_state

        episode_number += 1
        episode_rewards.append(episode_rew)
        cum_rew = sum(episode_rewards) / episode_number
        success = 1 * (episode_rew > SOLVED_THRESHOLD)
        cumulative_rewards.append(cum_rew)
        print(f"Episode: {episode_number} Reward: {episode_rew:.2f} Cumulative_Reward: {cum_rew:.2f} Success: {success}")
        file.writelines(f"Episode: {episode_number} Reward: {episode_rew:.2f} Cumulative_Reward: {cum_rew:.2f} Success: {success} \n")
        summation = sum(episode_rewards[episode_number - SUCCESS_INTERVAL:episode_number])
        if summation > SUCCESS_THRESHOLD * SUCCESS_INTERVAL and summation > max_rew * SUCCESS_INTERVAL:
            print("Agent Solved ", env_name, " environment")
            max_rew = summation/SUCCESS_INTERVAL
            torch.save(agent.Q_network.state_dict(), model_params_filename+str(episode_number))

    torch.save(agent.Q_network.state_dict(), model_params_filename)
    plt.title("DQN Agent Performance")
    plt.plot(episode_rewards)
    plt.xlabel("epoch")
    plt.ylabel("Reward")
    plt.savefig("DQN Agent Performance.png")
    plt.show()

    plt.title("DQN Agent Performance")
    plt.plot(cumulative_rewards)
    plt.xlabel("epoch")
    plt.ylabel("cumulative reward")
    plt.savefig("DQN Agent Performance Cumulative.png")
    plt.show()
else:
    agent.epsilon = 0
    success_test = 0
    agent.Q_network.load_state_dict(torch.load(model_params_filename))

    for i in range(TEST_EPISODE_NUMBER):
        reward = 0
        state = torch.tensor(env.reset())
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, r, done, _ = env.step(action.item())
            env.render()
            state = torch.tensor(next_state)
            reward += r
        success = 1 * (reward > SOLVED_THRESHOLD)
        success_test += success
        print(f"Episode Number: {i} Reward: {reward} Success: {success}")
    print(f"Success Rate: {success_test/TEST_EPISODE_NUMBER}")



