import numpy as np
import gym

# 創建環境
env = gym.make('FrozenLake-v0')

# 初始化Q表
Q = np.zeros((env.observation_space.n, env.action_space.n))

# 設置參數
alpha = 0.8
gamma = 0.95
epsilon = 0.1
episodes = 1000

# Q-Learning算法
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        next_state, reward, done, _ = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        state = next_state

print("Learned Q-Table:")
print(Q)