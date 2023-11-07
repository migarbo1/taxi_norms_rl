import numpy as np

class State():

    def __init__(self, x, y):
        self.pos = [x, y]
        self.client_on_board = 0
        self.view = np.zeros((8,))


    def to_array(self):
        return [*self.pos, self.client_on_board, *self.view]
    

    def update_car_view(self, grid):

        ul = grid[self.pos[0]-1, self.pos[1]-1] if self.pos[0] > 0 and self.pos[1] > 0 else -1
        u = grid[self.pos[0]-1, self.pos[1]] if self.pos[0] > 0 else -1
        ur = grid[self.pos[0]-1, self.pos[1]+1] if self.pos[0] > 0 and self.pos[1] < grid.shape[1]-1 else -1

        l = grid[self.pos[0], self.pos[1]-1] if self.pos[1] > 0 else -1
        r = grid[self.pos[0], self.pos[1]+1] if self.pos[1] < grid.shape[1]-1 else -1

        dl = grid[self.pos[0]+1, self.pos[1]-1] if self.pos[0] < grid.shape[0]-1 and self.pos[1] > 0 else -1
        d = grid[self.pos[0]+1, self.pos[1]] if self.pos[0] < grid.shape[0]-1 else -1
        dr = grid[self.pos[0]+1, self.pos[1]+1] if self.pos[0] < grid.shape[0]-1 and self.pos[1] < grid.shape[1]-1 else -1

        self.view = [ul, u, ur, l, r, dl, d, dr]

    
def array_to_state(array):
    state = State(*array[0:2])
    state.client_on_board = array[2]
    state.view = np.array(*array[3:])
    return state