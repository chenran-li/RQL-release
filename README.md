# Residual Q-Learning (RQL)
### [**[Project Page](https://sites.google.com/view/residualq-learning)**]  Official code base for **[Residual Q-Learning: Offline and Online Policy Customization without Value](https://arxiv.org/abs/2306.09526)**, ***NeurIPS 2023*** 

#



## Installation
### using pip
```
pip install swig cmake ffmpeg
pip install -r requirements.txt
```
### using Conda
```
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install swig cmake ffmpeg
pip install -r requirements.txt
```

## Training RL prior agents 
Run the script `train_rl.py` with arguments specifying the ennvironment, algorithm, and the number of expert rollouts collected in the final iteration which will be used for GAIL training. For example:
```
python train_rl.py --env-name highway_basic --algo dqn_ME --rollout-save-n-episodes 1000
python train_rl.py --env-name parking_basic --algo sac --rollout-save-n-episodes 10000
python train_rl.py --env-name cartpole_basic --algo dqn_ME --rollout-save-n-episodes 1000
python train_rl.py --env-name mountain_car_basic --algo sac --rollout-save-n-episodes 10000
```
The experiment will be logged in the directory in the format of `logs/{algo}/{env-name}_{timestamp}/`. The policies are stored in the subfolder `policies` and the final rollouts are stored in the subfolder `rollouts`. 

## Training IL prior agents
Run the script `train_gail.py` with arguments specifying the ennvironment, generator training algorithm, and the directory of the demo rollouts. For example:
```
python train_gail.py --env-name parking_basic --gen-algo sac --demo-rollout-path EXPERT_DIRECTORY/rollouts/final.npz
```

## Training Residual Q-learning customized agents
Run the script `train.py` with arguments specifying the environment, generator training algorithm, and prior model path. We provide pretrained RL prior agents for the environments we have tested. Run the following commands to train the customized agents with the pretrained RL prior agents. 
```
python -W ignore train.py --algo dqn_soft_residual --env highway-ME-basic-AddRightReward-v0 --prior-model-path ./logs/highway-ME-basic-v0_Example_Pretrained
python -W ignore train.py --algo sac_residual --env parking-basic-boundary-v0 --prior-model-path ./logs/parking-basic-v0_Example_Pretrained
python -W ignore train.py --algo dqn_soft_residual --env CartPole-modifed-morecenter-v1 --prior-model-path ./logs/CartPole-modifed-v1_Example_Pretrained 
python -W ignore train.py --algo sac_residual --env MountainCarContinuous-modifed-lessleft-v0 --prior-model-path ./logs/MountainCarContinuous-modifed-v0_Example_Pretrained
```

## Training RL full agents
Run the script `train_rl.py`. The environments correspond to the tasks with total rewards are specified by the named_config `cartpole_total`, `mountain_car_total`, `highway_total`, and `parking_total`.

## Highway Environment Demo Notebook
We also provide a demo notebook `Training_demo_Highway.ipynb` for the our experiments on the `highway-env` environment, which includes the training and evaluation of RL prior and customized policies. It also includes a demo of maximum-entropy MCTS for zero-shot online customization. 

### Citation
```
@inproceedings{li2023residual,
	title={Residual Q-Learning: Offline and Online Policy Customization without Value},
	author={Li, Chenran and Tang, Chen and Nishimura, Haruki and Mercat, Jean and Tomizuka, Masayoshi and Zhan, Wei},
	booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	year={2023}

}
```

## Questions
Please feel free to email us if you have any questions. 

Chenran Li ([chenran_li@berkeley.edu](mailto:chenran_li@berkeley.edu?subject=[GitHub]RQL-release))

Chen Tang ([chen_tang@berkeley.edu](mailto:chen_tang@berkeley.edu?subject=[GitHub]RQL-release))
