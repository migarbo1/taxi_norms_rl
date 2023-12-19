from replay_buffer import ReplayBuffer
from taxi_env.state import State
import torch.nn.functional as F
from taxi_env.env import Action
from network import DQN
import torch.nn as nn
import numpy as np
import random
import torch
import os


class DeepQLearning():

    def __init__(self, env, models_path, mas_training = False) -> None:
        
        self.env = env
        self.mas_training = mas_training

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        demo_state = State(0,0)
        self.num_features = len(demo_state.to_array())
        self.num_actions = len(Action)

        self.main_nn, self.trgt_nn, self.alt_trgt_nn = self.load_models(models_path)

        self.opt = torch.optim.Adam(self.main_nn.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        #hyperparameters
        self.episodes = 2000
        self.steps_per_episode = 256
        self.epsilon = 0.5
        self.epsilon_o = self.epsilon
        self.batch_size = 64
        self.discount = 0.99
        self.main_to_target_ratio = 2000
        self.save_ratio = 6000


    def load_models(self, path):
        model_name = self.get_model_name()
        main_nn = DQN(self.num_features, self.num_actions).to(self.device)
        trgt_nn = DQN(self.num_features, self.num_actions).to(self.device)

        if self.mas_training:
            alt_trgt_nn = DQN(self.num_features, self.num_actions).to(self.device)
            
        if os.path.exists(path):
            main_nn.load_state_dict(torch.load(f'{path}/{model_name}'))
            trgt_nn.load_state_dict(torch.load(f'{path}/{model_name}'))
            if self.mas_training:
                alt_trgt_nn.load_state_dict(trgt_nn.state_dict())


    def select_epsilon_greedy_action(self, state, inference = False):
        if inference:
            qs = self.trgt_nn(state).cpu().data.numpy()
            return np.argmax(qs)

        if np.random.uniform() < self.epsilon:
            return int(random.choice(list(Action)).value)
        else:
            qs = self.main_nn(state).cpu().data.numpy()
            return np.argmax(qs)


    def __train_step(self, states, actions, rewards, next_states):
        max_next_qs = self.trgt_nn(next_states).max(-1).values
        target = rewards + self.discount * max_next_qs
        qs = self.main_nn(states)
        action_masks = F.one_hot(actions, self.num_actions)
        masked_qs = (action_masks * qs).sum(dim=-1)
        loss = self.loss_fn(masked_qs, target.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


    def register_agents(self, num):
        states = []
        for _ in range(num):
            states.append(self.env.register_driver())
        return states


    def move_other_agent(self, state):
        torch_state = np.array(state.to_array()).astype(np.float32)
        torch_state = torch.from_numpy(np.expand_dims(torch_state, axis=0)).to(self.device)
        qs = self.alt_trgt_nn(torch_state).cpu().data.numpy()
        action = np.argmax(qs)
        reward, _state = self.env.step(state, action)
        return reward, _state


    def move_agents(self, states, rewards, jobs):
        new_states = []
        for i, state in enumerate(states):
            reward, _state = self.move_other_agent(self.env, state)
            new_states.append(_state)
            rewards[i] += reward
            jobs[i] += 1 if reward > 2 else 0
        return new_states


    def get_model_name(self):
        name = ''
        if self.env.is_normative:
            name += 'N'
        if self.mas_training:
            name += 'MAS'
        if name != '':
            name += '_'
        return name + 'target_nn.pth'
        


    def train(self):
        buffer = ReplayBuffer(100000, device=self.device)
        step_counter = 0
        losses = []

        #Training loop
        last_100_ep_rewards = []
        for episode in range(self.episodes+1):
            self.env.reset()

            o_state = self.env.register_driver()
            
            if self.mas_training:
                other_agent_state = self.env.register_driver()

            ep_reward = 0
            for i in range(self.steps_per_episode):

                if self.mas_training:
                    other_agent_state = self.move_other_agent(other_agent_state)
                    o_state.update_car_view(self.env.grid)

                state = np.array(o_state.to_array()).astype(np.float32)
                state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(self.device)
                action = self.select_epsilon_greedy_action(state_in)
                reward, o_next_state = self.env.step(o_state, action)
                next_state = np.array(o_next_state.to_array()).astype(np.float32)
                ep_reward += reward
                # Save to experience replay.
                buffer.add(state, action, reward, next_state)
                state = next_state
                o_state = o_next_state
                step_counter += 1
                # Copy main_nn weights to target_nn.
                if step_counter % self.main_to_target_ratio == 0:
                    self.trgt_nn.load_state_dict(self.main_nn.state_dict())

                if step_counter % self.save_ratio == 0:
                    model_name = self.get_model_name()
                    torch.save(self.trgt_nn.state_dict(), f'{os.getcwd()}/weights/{model_name}')
                    if self.mas_training:
                        self.alt_trgt_nn.load_state_dict(self.trgt_nn.state_dict())

                # Train neural network.
                if len(buffer) > self.batch_size:
                    states, actions, rewards, next_states = buffer.sample(self.batch_size)
                    loss = self.__train_step(states, actions, rewards, next_states)

            self.epsilon -= self.epsilon_o/(self.episodes+1)
            if len(last_100_ep_rewards) == 100:
                last_100_ep_rewards = last_100_ep_rewards[1:]
            last_100_ep_rewards.append(ep_reward)

            if episode % 50 == 0:
                print(f'Episode {episode}/{self.episodes}. Epsilon: {self.epsilon:.3f}.'
                    f' Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.2f}')
                
        torch.save(self.trgt_nn.state_dict(), f'{os.getcwd()}/weights/{model_name}')


    def eval(self, n_steps, n_agents):
        self.env.reset()
        o_state = self.env.register_driver()
        other_agent_states = self.register_agents(n_agents)
        other_agent_rewards = [0 for _ in range(len(other_agent_states))]
        other_agent_jobs = [0 for _ in range(len(other_agent_states))]
        ep_reward = 0
        n_jobs = 0
        for episode in range(n_steps):
            other_agent_states = self.move_agents(other_agent_states, other_agent_rewards, other_agent_jobs)
            o_state.update_car_view(self.env.grid)

            state = np.array(o_state.to_array()).astype(np.float32)
            state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(self.device)
            action = self.select_epsilon_greedy_action(state_in, True)
            reward, o_next_state = self.env.step(o_state, action)
            next_state = np.array(o_next_state.to_array()).astype(np.float32)
            ep_reward += reward
            if reward > 2:
                n_jobs += 1

            state = next_state
            o_state = o_next_state

        print(ep_reward)
        print(n_jobs)
        print(other_agent_rewards)
        print(other_agent_jobs)