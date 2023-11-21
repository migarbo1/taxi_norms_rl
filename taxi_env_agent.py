from spade.behaviour import CyclicBehaviour
from spade.template import Template
from spade.message import Message
from spade.agent import Agent
from state import SimpleState
from env import TaxiGridEnv
import json


class TaxiEnvAgent(Agent):

    def __init__(self, jid: str, password: str):
        super().__init__(jid, password)
        self.env = TaxiGridEnv()
        self.num_step = 0
        print('init finished')

    
    async def setup(self) -> None:
        template = Template(metadata={"performative": "REGISTER"})
        self.add_behaviour(self.RegisterDriverBehaviour(), template=template)
        print('register beh launched')
        template = Template(metadata={"performative": "STEP"})
        self.add_behaviour(self.PerformStepBehaviour(), template=template)
        print('perform step beh launched')


    class RegisterDriverBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            msg = await self.receive(timeout=2)
            if msg:
                print('register request received')
                state = self.agent.env.register_driver()
                msg = Message(to=str(msg.sender), body=json.dumps({'state': state.__dict__}), metadata={"performative": "REGISTERED"})
                await self.send(msg)
                print('confirmation sended')


    class PerformStepBehaviour(CyclicBehaviour):
        async def run(self) -> None:
            msg = await self.receive(timeout=2)
            if msg:
                print('request for step received')
                self.agent.num_step += 1
                body = json.loads(msg.body)
                reward, _state = self.agent.env.step(SimpleState.from_json(body['state']), int(body['action']))
                msg = Message(to=str(msg.sender), body=json.dumps({'state': _state.__dict__, 'reward': reward}), metadata={"performative": "STEP_RESPONSE"})
                await self.send(msg)
                print('response to step sended')
