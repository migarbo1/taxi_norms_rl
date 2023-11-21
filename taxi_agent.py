from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.template import Template
from spade.message import Message
from spade.agent import Agent
from state import SimpleState
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

    
    async def setup(self) -> None:
        self.add_behaviour(self.RequestRegisterBehaviour())
        template = Template(metadata={"performative": "STEP_RESPONSE"})
        self.add_behaviour(self.PerformStepBehaviour(), template=template)


    def decide_action(self):
        qs = self.q_table.get(self.state, np.zeros(NUM_ACTIONS,))
        print(qs)
        if self.inference:
            return np.argmax(qs)
        else:
            if random.uniform(0, 1) > EPSILON:
                return np.argmax(qs)
            else:
                return random.randint(0, NUM_ACTIONS-1)


    def update_q_table(self, action, _state, reward):
        if not self.inference:
            qs = self.q_table.get(self.state, np.zeros(NUM_ACTIONS,))
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
            print('register petititon sended')
            resp = None
            while not resp:
                resp = await self.receive(timeout=2)
            print('register response received')
            body = json.loads(resp.body)
            self.agent.state = SimpleState.from_json(body['state'])


    class PerformStepBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            if not self.agent.state:
                return 
            action = self.agent.decide_action()
            msg = Message(to=str(self.agent.env_jid), body=json.dumps({'state': self.agent.state.__dict__, 'action': int(action)}), metadata={"performative": "STEP"})
            await self.send(msg)
            print('petition for step sended')
            resp = None
            while not resp:
                resp = await self.receive(timeout=2)
            print('response to step received')
            body = json.loads(resp.body)
            _state, reward = SimpleState.from_json(body['state']), body['reward']
            self.agent.update_q_table(action, _state, reward)
