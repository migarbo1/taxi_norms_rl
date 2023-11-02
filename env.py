import numpy as np
from enum import Enum
import random
from state import State

class GridZones(Enum):
    BARRIERS = -1
    QUEUE = 1
    CAR = 3
    PICKUP = 10
    DROP1 = 100
    DROP2 = 75
    DROP3 = 50


class Action(Enum):
    WAIT = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    PICK = 2
    DROP = 2


class TaxiGridEnv():

    def __init__(self):
        self.grid = np.zeros((8,12))
        self.drivers_dict = {}
        self._setup_grid()
        print(self.grid)

    def _setup_grid(self):
        self._set_queue()
        self._set_pickup()
        self._set_drop_zones()
        self._set_barriers()

    def _set_queue(self):
        self.grid[1:-1, 0] = GridZones.QUEUE.value

    def _set_pickup(self):
        self.grid[-1, 0] = GridZones.PICKUP.value

    def _set_drop_zones(self):
        self.grid[-1, 11] = GridZones.DROP1.value
        self.grid[-1, 6] = GridZones.DROP2.value
        self.grid[3, 7] = GridZones.DROP3.value

    def _set_barriers(self):
        self.grid[1:4, 2] = GridZones.BARRIERS.value
        self.grid[5:7, 2] = GridZones.BARRIERS.value

        self.grid[1, 5:9] = GridZones.BARRIERS.value
        self.grid[3, 4:6] = GridZones.BARRIERS.value

        self.grid[5, -3:] = GridZones.BARRIERS.value

        self.grid[1:3, -1] = GridZones.BARRIERS.value

        self.grid[6:, 5] = GridZones.BARRIERS.value

    def register_driver(self):
        pos_code = GridZones.BARRIERS.value
        car_in_pos = True

        while pos_code != 0 and car_in_pos:
            x = random.randint(0, self.grid.shape[0] - 1)
            y = random.randint(0, self.grid.shape[1] - 1)
            pos_code = self.grid[x,y]
            car_in_pos = False or (self.drivers_dict[k] == (x, y) for k in self.drivers_dict.keys())
        
        self.drivers_dict[len(self.drivers_dict.keys())] = (x, y)
        return State(x, y)

    def step(self, state, action):
        pass

env = TaxiGridEnv()
state = env.register_driver()
print(state.to_array())