import numpy as np
import random
from taxi_env.env import TaxiGridEnv
from utils import *
from tqdm import tqdm

EPISODES = 50000
EPISODE_STEP_LIMIT = 500
LR = 0.1 
GAMMA = 0.99
EPSILON = 0.1
NUM_ACTIONS = 5

def select_max_action(qs):
    return np.argmax(qs)


def select_action(qs):
    if random.uniform(0, 1) > EPSILON:
        action =  select_max_action(qs)
        return action
    else:
        return random.randint(0, NUM_ACTIONS-1)


def reset(env: TaxiGridEnv):
    env.reset()
    state = env.register_driver()

    return state


def q_learning(Q, N, env):

    for i in tqdm(range(EPISODES)):
        steps_in_episode = 0
        state = reset(env)
        while steps_in_episode < EPISODE_STEP_LIMIT:
            state_count = N.get(state, 0)
            N[state] = state_count + 1

            qs = Q.get(state, np.zeros(NUM_ACTIONS,))
            action = select_action(qs)

            reward, new_state = env.step(state, action)
            qs_ = Q.get(new_state, np.zeros(NUM_ACTIONS,))

            delta = qs[action] + LR * (reward + GAMMA * qs_[select_max_action(qs_)] - qs[action])
            qs[action] = delta
            Q[state] = qs
            state = new_state
            steps_in_episode += 1

    save_object('.', 'Q', Q)
    save_object('.', 'N', N)
    

if __name__ == '__main__':
    Q = {}
    N = {}
    env = TaxiGridEnv()
    if os.path.exists('./Q1M.pickle'):
        Q =  load_object('.','Q1M')
    q_learning(Q,N,env)