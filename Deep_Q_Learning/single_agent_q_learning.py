from taxi_env.env import TaxiGridEnv, Action
from replay_buffer import ReplayBuffer
from taxi_env.state import State
import torch.nn.functional as F
from network import DQN
import torch.nn as nn
import numpy as np
import random
import torch
import math
import os


episodes = 1000
steps_per_episode = 150
epsilon = 0.05
batch_size = 256
discount = 0.99


def select_epsilon_greedy_action(state, epsilon, inference = False):
    if inference:
        qs = trgt_nn(state).cpu().data.numpy()
        return np.argmax(qs)

    if np.random.uniform() < epsilon:
        return int(random.choice(list(Action)).value)
    else:
        qs = main_nn(state).cpu().data.numpy()
        return np.argmax(qs)


def train_step(states, actions, rewards, next_states):
    max_next_qs = trgt_nn(next_states).max(-1).values
    target = rewards + discount * max_next_qs
    qs = main_nn(states)
    action_masks = F.one_hot(actions, num_actions)
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_fn(masked_qs, target.detach())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


env = TaxiGridEnv()
demo_state = State(0,0)
num_features = len(demo_state.to_array())
num_actions = len(Action)
print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_nn = DQN(num_features, num_actions).to(device)
trgt_nn = DQN(num_features, num_actions).to(device)
if os.path.exists(f'{os.getcwd()}/weights/SA_target_nn.pth'):
    if not torch.cuda.is_available():
        main_nn.load_state_dict(torch.load(f'{os.getcwd()}/weights/SA_target_nn.pth', map_location=torch.device('cpu')))
        trgt_nn.load_state_dict(torch.load(f'{os.getcwd()}/weights/SA_target_nn.pth', map_location=torch.device('cpu')))
    else:
        main_nn.load_state_dict(torch.load(f'{os.getcwd()}/weights/SA_target_nn.pth'))
        trgt_nn.load_state_dict(torch.load(f'{os.getcwd()}/weights/SA_target_nn.pth'))

opt = torch.optim.Adam(main_nn.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

buffer = ReplayBuffer(100000, device=device)
step_counter = 0
losses = []

#Training loop
last_100_ep_rewards = []
for episode in range(episodes+1):
  env.reset()
  o_state = env.register_driver()
  state = np.array(o_state.to_array()).astype(np.float32)
  ep_reward = 0
  for i in range(steps_per_episode):
    state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
    action = select_epsilon_greedy_action(state_in, epsilon)
    reward, o_next_state = env.step(o_state, action)
    next_state = np.array(o_next_state.to_array()).astype(np.float32)
    ep_reward += reward
    # Save to experience replay.
    buffer.add(state, action, reward, next_state)
    state = next_state
    o_state = o_next_state
    step_counter += 1
    # Copy main_nn weights to target_nn.
    if step_counter % 2000 == 0:
      trgt_nn.load_state_dict(main_nn.state_dict())
    
    # Train neural network.
    if len(buffer) > batch_size:
      states, actions, rewards, next_states = buffer.sample(batch_size)
      loss = train_step(states, actions, rewards, next_states)


  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 50 == 0:
    print(f'Episode {episode}/{episodes}. Epsilon: {epsilon:.3f}.'
          f' Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.2f}')