import matplotlib.pyplot as plt

def plotting(file_name, env_name):
    file = open(file_name, 'r')
    avg_cum_reward = []
    for row in file:
        avg_cum_reward.append(float(row.split('average reward:')[1].split('elapsed')[0]))
    plt.plot(avg_cum_reward)
    plt.title("Performance in "+env_name+" environment")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Average Reward")
    return
