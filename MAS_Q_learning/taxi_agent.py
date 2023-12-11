import asyncio
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.template import Template
from spade.message import Message
from spade.agent import Agent
from taxi_env.state import State, SimpleState
from utils.utils import *
import numpy as np
import random
import json

LR = 0.1 
GAMMA = 0.99
EPSILON = 0.1
NUM_ACTIONS = 5

class TaxiAgent(Agent):

    def __init__(self, jid: str, password: str, env_jid: str, q_table, inference: bool = False):
        super().__init__(jid, password)
        self.state = None
        self.env_jid = env_jid
        self.inference = inference
        self.q_table = q_table
        self.total_reward = 0
        self.jobs_completed = 0
        self.simple_q_table = load_object('.','Q1M_SimpleState') if os.path.exists('./Q1M_SimpleState.pickle') else {}

    
    async def setup(self) -> None:
        self.add_behaviour(self.RequestRegisterBehaviour())
        template = Template(metadata={"performative": "STEP_RESPONSE"})
        self.add_behaviour(self.PerformStepBehaviour(), template=template)
        template = Template(metadata={"performative": "RESET"})
        self.add_behaviour(self.ResetBehaviour(), template=template)


    def get_qs(self):
        qs = self.q_table.get(self.state, np.zeros(NUM_ACTIONS,))
        if sum(qs) == 0:
            simple_state = SimpleState(*self.state.pos)
            simple_state.client_on_board = self.state.client_on_board
            idx = [1,6,3,4] #up, down, left, right
            simple_state.view = list(np.array(self.state.view)[idx])
            if 3 not in simple_state.view:
                qs = self.simple_q_table.get(simple_state, np.zeros(NUM_ACTIONS,))
        return qs



    def decide_action(self):
        qs = self.get_qs()

        print(f'{self.jid} | state: {self.state} | q table: {qs}')

        if self.inference:
            return np.argmax(qs)
        else:
            if random.uniform(0, 1) > EPSILON:
                return np.argmax(qs)
            else:
                return random.randint(0, NUM_ACTIONS-1)




    def update_q_table(self, action, _state, reward):
        if not self.inference:
            qs = self.get_qs()
            _qs = self.q_table.get(_state, np.zeros(NUM_ACTIONS,))

            delta = qs[action] + LR * (reward + GAMMA * _qs[np.argmax(_qs)] - qs[action])
            qs[action] = delta
            self.q_table[self.state] = qs

        self.state = _state
        self.total_reward += reward

        if reward > 10:
            self.jobs_completed += 1


    class RequestRegisterBehaviour(OneShotBehaviour):
        async def run(self) -> None:
            msg = Message(to=str(self.agent.env_jid), body=json.dumps({}), metadata={"performative": "REGISTER"})
            await self.send(msg)
            
            resp = None
            while not resp:
                resp = await self.receive(timeout=2)
            
            body = json.loads(resp.body)
            self.agent.state = State.from_json(body['state'])


    class PerformStepBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            if not self.agent.state:
                return 
            
            action = self.agent.decide_action()
            msg = Message(to=str(self.agent.env_jid), body=json.dumps({'state': self.agent.state.__dict__, 'action': int(action)}), metadata={"performative": "STEP"})
            await self.send(msg)
            
            resp = None
            count = 0
            while not resp and count < 3:
                await asyncio.sleep(0.1)
                resp = await self.receive(timeout=1)
                count += 1
            
            if resp:
                body = json.loads(resp.body)
                if body['ood']:
                    self.agent.state = State.from_json(body['state'])
                else:
                    _state, reward = State.from_json(body['state']), body['reward']
                    self.agent.update_q_table(action, _state, reward)


    class ResetBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            msg = await self.receive(timeout=2)
            if msg:
                body = json.loads(msg.body)
                _state = State.from_json(body['state'])
                self.agent.state = _state
