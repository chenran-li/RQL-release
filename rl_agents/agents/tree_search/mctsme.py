import logging
import numpy as np
import torch as th
import scipy
import scipy.special
from functools import partial

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.olop import OLOP

logger = logging.getLogger(__name__)


class MCTSAgentME(AbstractTreeSearchAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def make_planner(self):
        prior_policy = MCTSAgentME.policy_factory(self.config["prior_policy"])
        rollout_policy = MCTSAgentME.policy_factory(self.config["rollout_policy"])
        return MCTSME(self.env, prior_policy, rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "horizon": None,
            "prior_policy": {"type": "random"},
            "rollout_policy": {"type": "random"},
            "env_preprocessors": []
         })
        return config

    @staticmethod
    def policy_factory(policy_config):
        if policy_config["type"] == "random":
            return MCTSAgentME.random_policy
        elif policy_config["type"] == "random_available":
            return MCTSAgentME.random_available_policy
        elif policy_config["type"] == "preference":
            return partial(MCTSAgentME.preference_policy,
                           action_index=policy_config["action"],
                           ratio=policy_config["ratio"])
        else:
            raise ValueError("Unknown policy type")

    @staticmethod
    def random_policy(state, observation):
        """
            Choose actions from a uniform distribution.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(actions))) / len(actions)
        return actions, probabilities

    @staticmethod
    def random_available_policy(state, observation):
        """
            Choose actions from a uniform distribution over currently available actions only.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

    @staticmethod
    def preference_policy(state, observation, action_index, ratio=2):
        """
            Choose actions with a distribution over currently available actions that favors a preferred action.

            The preferred action probability is higher than others with a given ratio, and the distribution is uniform
            over the non-preferred available actions.
        :param state: the environment state
        :param observation: the corresponding observation
        :param action_index: the label of the preferred action
        :param ratio: the ratio between the preferred action probability and the other available actions probabilities
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        for i in range(len(available_actions)):
            if available_actions[i] == action_index:
                probabilities = np.ones((len(available_actions))) / (len(available_actions) - 1 + ratio)
                probabilities[i] *= ratio
                return available_actions, probabilities
        return MCTSAgentME.random_available_policy(state, observation)


class MCTSME(AbstractPlanner):
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """
            New MCTSME instance.

        :param config: the mcts configuration. Use default if None.
        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                OLOP.allocation(self.config["budget"], self.config["gamma"])
        print(self.config["episodes"],self.config["horizon"])

        if (self.config["DQN_prior"]==1):
            from stable_baselines3 import DQN_ME
            model_path = self.config["model_path"]
            self.DQN_model = DQN_ME.load(model_path,device="cpu")
            self.prior_policy = self.DQN_prior_policy
            self.rollout_policy = self.DQN_prior_policy
            print("using priror policy")

    def DQN_prior_policy(self, state, observation):
        new_obs = np.array([observation])
        Q_value = self.DQN_model.q_net(th.from_numpy(new_obs))
        p_soft = th.exp(Q_value - (th.logsumexp(Q_value,1).reshape(-1,1)).expand(-1, Q_value.shape[1]))
        actions = np.arange(state.action_space.n)
        probabilities = ((p_soft).detach().numpy()).flatten()
        return actions, probabilities


    @classmethod
    def default_config(cls):
        cfg = super(MCTSME, cls).default_config()
        cfg.update({
            "temperature": 1,
            "closed_loop": False
        })
        print("temperature", cfg["temperature"])
        return cfg

    def reset(self):
        self.root = MCTSNode(parent=None, planner=self)

    def run(self, state, observation):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        node = self.root
        total_reward = 0
        depth = 0
        terminal = False
        while depth < self.config['horizon'] and node.children and not terminal:
            action = node.sampling_rule(temperature=self.config['temperature']) # selection
            observation, reward, terminal, _ = self.step(state, action) # env move one step
            node = node.get_child(action) # get next node
            node.action_reward = reward # update the action reward
            depth += 1

        if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))

        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)
        node.update_branch(total_reward)

    def evaluate(self, state, observation, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param observation: the corresponding observation.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        for h in range(0, self.config["horizon"] - depth):
            actions, probabilities = self.rollout_policy(state, observation)
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** h * reward
            if np.all(terminal):
                break
        return total_reward

    def plan(self, state, observation):
        self.root.count -= 1
        for i in range(self.config['episodes']):
            self.run(safe_deepcopy_env(state), observation)
        return self.get_plan()

    def get_plan(self):
        """
            Get the optimal action sequence of the current tree by recursively selecting the best action within each
            node with no exploration.

        :return: the list of actions
        """
        actions = []
        node = self.root
        while len(actions) < 1:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions


class MCTSNode(Node):
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner, prior=1):
        super(MCTSNode, self).__init__(parent, planner)
        self.value = 0
        self.prior = prior
        self.action_reward = 0

    def selection_rule(self):
        if not self.children:
            return None

        # Tie best counts by best value    
        all_childrens = list(self.children.values())
        all_q_values = np.array([child.value for child in all_childrens])
        all_pre_values = np.array([np.log(child.prior) for child in all_childrens])
        log_pro = all_q_values + all_pre_values 
        idx = np.argmax(log_pro)
        return idx

    def sampling_rule(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """

        if self.children:
            all_childrens = list(self.children.values())
            a_dim = len(all_childrens)
            if (self.count < a_dim):
                return self.count

            tem = (temperature * a_dim) / np.log(1+self.count)

            average_pro = np.ones(a_dim) / a_dim
            
            if tem < 1:
                all_q_values = np.array([child.value for child in all_childrens])
                expqsum = scipy.special.logsumexp(all_q_values)
                log_pro = all_q_values - expqsum
                pro = np.exp(log_pro)
                pro = (1 - tem) * pro + tem * average_pro
            else:
                pro = average_pro
            pro = np.cumsum(pro)
            p_sample = 0.999999*np.random.rand(1)
            idx = np.searchsorted(pro, p_sample)
            return idx[0]
        else:
            return None

    def expand(self, actions_distribution):
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value = self.action_reward + self.planner.config["gamma"] * total_reward

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            all_childrens = list(self.parent.children.values())
            all_q_values = np.array([child.value for child in all_childrens])
            all_pre_values = np.array([np.log(child.prior) for child in all_childrens])
            logexpqsum = scipy.special.logsumexp(all_q_values + all_pre_values)

            self.parent.update_branch(logexpqsum)
        elif (self.planner.config["Plot"]==1):
            all_childrens = list(self.children.values())
            all_q_values = np.array([child.value for child in all_childrens])
            all_pre_values = np.array([np.log(child.prior) for child in all_childrens])
            logexpqsum = scipy.special.logsumexp(all_q_values + all_pre_values)
            log_pro = all_q_values + all_pre_values - logexpqsum
            pro = np.exp(log_pro)
            #print(pro)

    def get_child(self, action):
        child = self.children[action]
        return child

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """
            For any node in the subtree, convert the distribution of all children visit counts to prior
            probabilities, and reset the visit counts.

        :param regularization: in [0, 1], used to add some probability mass to all children.
                               when 0, the prior is a Boltzmann distribution of visit counts
                               when 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = (1 - regularization)*(child.count+1)/total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

    def get_value(self):
        return self.value