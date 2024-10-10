import numpy as np
import random
import copy


# by default, actions are ordered
# [left, up, down, right]
def get_cliff_mdp():
    pass


# by default, actions are ordered
# [left, up, down, right]
def get_optimal_policy():
    pass


def get_pi_e_greedy(pi, epsilon):
    pass


def get_P_R(MDP, pi):
    pass


def get_v(P, R, gamma):
    pass


# Generate a full trajectory under the policy pi.
# The trajectory must start in init_state and end in terminal
def gen_episode(MDP, pi, num_actions, init_state, terminal):
    pass


if __name__ == "__main__":

    random.seed(1)

    gamma = 0.9

    cliff_mdp = get_cliff_mdp()

    pi = get_optimal_policy()

    # YOUR CODE GOES HERE