import pathlib
import warnings

import os
import datetime
import argparse
import highway_env
from sacred.observers import FileStorageObserver

import rl_zoo3.import_envs
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.scripts.config.train_adversarial import train_adversarial_ex
from imitation.scripts.train_adversarial import train_adversarial
warnings.filterwarnings('ignore')

@train_adversarial_ex.main
def train_gail():
    return train_adversarial(algo_cls=gail_algo.GAIL)

if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env-name', type=str, default='')
    parser.add_argument('--gen-algo', type=str, default='sac') # sac, ppo, or dqn_ME
    parser.add_argument('--demo-rollout-path', type=str, default='')
    parser.add_argument('--total-timesteps', type=int, default=-1) 
    parser.add_argument('--checkpoint-interval', type=int, default=500)
    parser.add_argument('--agent-path', type=str, default='')
    args = parser.parse_args()

    if args.env_name == '':
        raise ValueError('Please specify the environment.')

    now = datetime.datetime.now()
    timestamp = now.isoformat()
    logdir = os.path.join('logs', 'gail', args.env_name + '_{}/'.format(timestamp))

    config_updates = {
            'seed': args.seed,
            'logging.log_dir': logdir,
            'demonstrations.rollout_path': args.demo_rollout_path,
            'checkpoint_interval': args.checkpoint_interval,
        }
    if args.agent_path != '':
        config_updates['agent_path'] = args.agent_path
    if args.total_timesteps != -1:
        config_updates['total_timesteps'] = args.total_timesteps
        
    observer_path = pathlib.Path.cwd() / "logs" / "sacred" / "train_gail"
    observer = FileStorageObserver(observer_path)
    train_adversarial_ex.observers.append(observer)
    train_adversarial_ex.run(
        named_configs=[
            args.env_name, # environment named_config
            f'rl.{args.gen_algo}', # rl algorithm named_config
            f'policy.{args.gen_algo}' # policy named_config
        ],
        config_updates=config_updates,
    )