from spade.behaviour import CyclicBehaviour
from spade.template import Template
from spade.message import Message
from spade.agent import Agent
from state import SimpleState
from env import TaxiGridEnv
import json


MAX_STEPS_IN_EPISODE = 1000 # 2' 15" aprox per episode
MAX_EPISODES = 100


class TaxiEnvAgent(Agent):

    def __init__(self, jid: str, password: str):
        super().__init__(jid, password)
        self.env = TaxiGridEnv()
        self.num_step = 0
        self.finished = False
        self.agents = []

    
    async def setup(self) -> None:
        template = Template(metadata={"performative": "REGISTER"})
        self.add_behaviour(self.RegisterDriverBehaviour(), template=template)
        template = Template(metadata={"performative": "STEP"})
        self.add_behaviour(self.PerformStepBehaviour(), template=template)


    class RegisterDriverBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            msg = await self.receive(timeout=2)
            if msg:
                self.agent.agents.append(str(msg.sender))
                state = self.agent.env.register_driver()
                msg = Message(to=str(msg.sender), body=json.dumps({'state': state.__dict__}), metadata={"performative": "REGISTERED"})
                await self.send(msg)


    class PerformStepBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            msg = await self.receive(timeout=2)
            if msg:
                self.agent.num_step += 1
                body = json.loads(msg.body)
                reward, _state = self.agent.env.step(SimpleState.from_json(body['state']), int(body['action']))
                msg = Message(to=str(msg.sender), body=json.dumps({'state': _state.__dict__, 'reward': reward}), metadata={"performative": "STEP_RESPONSE"})
                await self.send(msg)

                if self.agent.num_step == MAX_EPISODES * MAX_STEPS_IN_EPISODE:
                    self.agent.finished = True

                if self.agent.num_step % MAX_STEPS_IN_EPISODE == 0:
                    print(f'{self.agent.num_step/MAX_STEPS_IN_EPISODE} episodes of {MAX_EPISODES}')
                    await self.launch_reset()
        
        
        async def launch_reset(self):
            self.agent.env = TaxiGridEnv()
            for agent in self.agent.agents:
                state = self.agent.env.register_driver()
                msg = Message(to=agent, body=json.dumps({'state': state.__dict__}), metadata={"performative": "RESET"})
                await self.send(msg)