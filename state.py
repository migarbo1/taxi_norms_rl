import numpy as np

class State():

    def __init__(self, x, y):
        self.pos = [x, y]
        self.client_on_board = 0
        self.view = np.zeros((8,))

    def to_array(self):
        return [*self.pos, self.client_on_board, *self.view]
    
def array_to_state(array):
    state = State(*array[0:2])
    state.client_on_board = array[2]
    state.view = np.array(*array[3:])
    return state