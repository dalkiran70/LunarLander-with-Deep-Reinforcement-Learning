import matplotlib.pyplot as plt
file = open("DQN_logs.txt", 'r')


avg_cum_reward = []
avg_rew = []
for row in file:
    first = row.split('Reward:')[1]
    avg_rew.append(float(first.split('Cumulative_')[0]))

plt.plot(avg_rew)
plt.ylim([-650, +300])
plt.title("Performance LunarLander-v2 environment")
plt.xlabel("Episode")
plt.ylabel("Reward per Episode")
plt.show()

