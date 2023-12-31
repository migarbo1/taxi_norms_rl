from taxi_env_agent import TaxiEnvAgent
from taxi_agent import TaxiAgent
from utils.utils import *
import asyncio
import spade
import tqdm


def merge_agents_q(t1, t2):
    Q = {}
    common_keys = set(t1.keys()).intersection(set(t2.keys()))
    for k in common_keys:
        Q[k] = t1[k] + t2[k]
    
    keys_only_in_1 = set(t1.keys()) - set(t2.keys())

    for k in list(keys_only_in_1):
        Q[k] = t1[k]

    keys_only_in_2 = set(t2.keys()) - set(t1.keys())
    for k in list(keys_only_in_2):
        Q[k] = t2[k]
    return Q


async def main(env_agent, taxi_agent_1, taxi_agent_2):
    await env_agent.start()
    await asyncio.sleep(0.5)
    await taxi_agent_1.start()
    await taxi_agent_2.start()

    while not env_agent.finished:
        await asyncio.sleep(0.1)     



if __name__ == '__main__':
    try:
        # Q = load_object('.','Q1M_CompleteState') if os.path.exists('./Q1M_CompleteState.pickle') else {}
        Q = load_object('.','Q/Q_MAS_CompleteState_aux') if os.path.exists('./Q_MAS_CompleteState_aux.pickle') else {}

        env_agent = TaxiEnvAgent('taxi_env_agent@gtirouter.dsic.upv.es', 'pass')
        taxi_agent_1 = TaxiAgent('taxi_agent_1@gtirouter.dsic.upv.es', 'pass', 'taxi_env_agent@gtirouter.dsic.upv.es', Q, inference=True)
        taxi_agent_2 = TaxiAgent('taxi_agent_2@gtirouter.dsic.upv.es', 'pass', 'taxi_env_agent@gtirouter.dsic.upv.es', Q, inference=False)
              
        spade.run(main(env_agent, taxi_agent_1, taxi_agent_2))

        print('agent_total_reward: ', taxi_agent_1.total_reward)
        print('agent_total_reward: ', taxi_agent_2.total_reward)
        print('jobs completed: ', taxi_agent_1.jobs_completed)
        print('jobs completed: ', taxi_agent_2.jobs_completed)
    except:
        # Q = taxi_agent_1.q_table if taxi_agent_1.jobs_completed > taxi_agent_2.jobs_completed else taxi_agent_2.q_table
        Q = taxi_agent_2.q_table
        save_object('.', 'Q_MAS_CompleteState_aux', Q)
        print('agent_total_reward: ', taxi_agent_1.total_reward)
        print('agent_total_reward: ', taxi_agent_2.total_reward)
        print('jobs completed: ', taxi_agent_1.jobs_completed)
        print('jobs completed: ', taxi_agent_2.jobs_completed)
    # Q = taxi_agent_1.q_table if taxi_agent_1.jobs_completed > taxi_agent_2.jobs_completed else taxi_agent_2.q_table
    Q = taxi_agent_2.q_table
    
    save_object('.', 'Q_MAS_CompleteState_aux', Q)
    exit()
