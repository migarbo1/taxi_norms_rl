from env import TaxiGridEnv, Action
import random
import numpy as np
from utils import *
import time
from state import SimpleState

STEPS = 100
NUM_ACTIONS = 5
EPSILON = 0.1

def select_max_action(qs):
    return np.argmax(qs)


def select_action(qs, deterministic = True):
    if deterministic:
        return select_max_action(qs)
    else:
        if random.uniform(0, 1) > EPSILON:
            return select_max_action(qs)
        else:
            return random.randint(0, NUM_ACTIONS-1)
    

def reset(env: TaxiGridEnv):
    env.reset()
    state = env.register_driver()

    return state

def create_q_from_state_to_simple_state(Q):
    Q_ = {}
    for k in Q.keys():
        val = Q[k]
        k_ = SimpleState(*k.pos)
        k_.client_on_board = k.client_on_board
        k_.view = [k.view[1], k.view[6], k.view[3], k.view[4]]
        Q_[k_] = val

    return Q_


if __name__ == '__main__':
    env = TaxiGridEnv()
    state = reset(env)
    Q = load_object('.','Q1M_SimpleState')
    cummulative_reward = 0
    jobs_completed = 0
    for _ in range(STEPS):
        print(state)
        qs = Q.get(state, np.zeros(NUM_ACTIONS,))
        print(qs)
        action = select_action(qs)
        reward, new_state = env.step(state, action)
        state = new_state
        cummulative_reward += reward
        print(f'action: {Action(action)}, reward: {reward}\nGrid:\n{env.grid}')
        if reward > 10:
            jobs_completed += 1
    print(f'cummulative_reward: {cummulative_reward}\nJobs completed: {jobs_completed}')

    # Q = load_object('.','Q1M_State')
    # Q_ = create_q_from_state_to_simple_state(Q)
    # save_object('.', 'Q1M_SimpleState', Q_)

    