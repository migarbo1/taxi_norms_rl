from utils import *
from state import *
from env import TaxiGridEnv

if __name__ == '__main__':
    Q = load_object('.','Q1M_SimpleState') if os.path.exists('./Q1M_SimpleState.pickle') else {}
    new_q = {}
    env = TaxiGridEnv()
    for simple_state, value in Q.items():
        complete_state = State(*simple_state.pos)
        complete_state.update_car_view(env.grid)
        complete_state.client_on_board = simple_state.client_on_board
        new_q[complete_state] = value
    save_object('.', 'Q1M_CompleteState', new_q)