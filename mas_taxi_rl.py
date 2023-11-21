from taxi_env_agent import TaxiEnvAgent
from taxi_agent import TaxiAgent
from utils import *
import asyncio
import spade

MAX_STEPS = 100


async def main():
    Q = load_object('.','Q1M_SimpleState') if os.path.exists('./Q1M_SimpleState.pickle') else {}
    env_agent = TaxiEnvAgent('taxi_env_agent@your.xmpp.server', 'pass')
    taxi_agent = TaxiAgent('taxi_agent_1@your.xmpp.server', 'pass', 'taxi_env_agent@your.xmpp.server', Q, inference=True)
    await env_agent.start()
    await asyncio.sleep(0.5)
    await taxi_agent.start()

    while env_agent.num_step < MAX_STEPS:
        await asyncio.sleep(0.1)

    print('agent_total_reward: ', taxi_agent.total_reward)
    print('jobs completed: ', taxi_agent.jobs_completed)
    exit()


if __name__ == '__main__':
    spade.run(main())
