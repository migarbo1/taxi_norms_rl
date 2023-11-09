from env import TaxiGridEnv, Action
import random
import numpy as np
from utils import load_object

STEPS = 1000
NUM_ACTIONS = 5
EPSILON = 0.1

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


if __name__ == '__main__':
    env = TaxiGridEnv()
    state = reset(env)
    Q = load_object('.','Q')
    print(len(Q.keys()))
    for s in list(Q.keys()):
        print(s)
    cummulative_reward = 0
    for _ in range(STEPS):
        print(state)
        qs = Q.get(state, np.zeros(NUM_ACTIONS,))
        print(qs)
        action = select_action(qs)
        reward, new_state = env.step(state, action)
        state = new_state
        cummulative_reward += reward
        print(f'action: {Action(action)}, reward: {reward}\nGrid:\n{env.grid}')
    print(f'cummulative_reward: {cummulative_reward}')