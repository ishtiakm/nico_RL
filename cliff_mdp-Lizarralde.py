import numpy as np
import random
import copy


# by default, actions are ordered
# [left, up, down, right]
def get_cliff_mdp(k_states = 38): #If you hit the border, you remain in the same cell and receive a reward of -1.
    """
    The function return two 3D arrays: (4 x 38 x 38), corresponding to the 4 possible actions and 38 states.
    First array is the matrices of probability.
    Second array is the matrices of rewards.
    By default, actions are ordered [left, up, down, right]
    """

    P_left = np.zeros((k_states,k_states))
    P_up = np.zeros((k_states,k_states))
    P_down = np.zeros((k_states,k_states))
    P_right = np.zeros((k_states,k_states))

    R_left = np.full((k_states,k_states), -1)
    R_up = np.full((k_states,k_states), -1)
    R_down = np.full((k_states,k_states), -1)
    R_right = np.full((k_states,k_states), -1)

    # left
    for state_src_index in range(k_states):
        state_src_label = state_src_index + 1
        R_dest = -1

        if state_src_label == 2: # goal
            state_dest_label = 2
            R_dest = 0
        elif state_src_label <= 5: # wall -> stay
            state_dest_label = state_src_label
        else: # move
            state_dest_label = state_src_label - 3

        state_dest_index = state_dest_label - 1

        P_left[state_src_index,state_dest_index] = 1
        R_left[state_src_index,state_dest_index] = R_dest

    # up
    for state_src_index in range(k_states):
        state_src_label = state_src_index + 1
        R_dest = -1

        if state_src_label == 2: # goal
            state_dest_label = 2
            R_dest = 0
        elif state_src_label == 1: # move
            state_dest_label = 5
        elif state_src_label % 3 == 0: # wall -> stay
            state_dest_label = state_src_label
        else: # move
            state_dest_label = state_src_label - 1

        state_dest_index = state_dest_label - 1

        P_up[state_src_index,state_dest_index] = 1
        R_up[state_src_index,state_dest_index] = R_dest

    # down
    for state_src_index in range(k_states):
        state_src_label = state_src_index + 1
        R_dest = -1

        if state_src_label == 2: # goal
            state_dest_label = 2
            R_dest = 0
        elif state_src_label == 1: # wall -> stay
            state_dest_label = state_src_label
        elif state_src_label == 5: # move
            state_dest_label = 1
        elif state_src_label == k_states: # move
            state_dest_label = 2
        elif (state_src_label+1) % 3 == 0: # cliff
            state_dest_label = 1
            R_dest = -100
        else: # move
            state_dest_label = state_src_label + 1

        state_dest_index = state_dest_label - 1

        P_down[state_src_index,state_dest_index] = 1
        R_down[state_src_index,state_dest_index] = R_dest

    # right
    for state_src_index in range(k_states):
        state_src_label = state_src_index + 1
        R_dest = -1

        if state_src_label == 2: # goal
            state_dest_label = 2
            R_dest = 0
        elif state_src_label == 1: # cliff
            state_dest_label = 1
            R_dest = -100
        elif state_src_label >= k_states-2: # wall -> stay
            state_dest_label = state_src_label
        else: # move
            state_dest_label = state_src_label + 3

        state_dest_index = state_dest_label - 1

        P_right[state_src_index,state_dest_index] = 1
        R_right[state_src_index,state_dest_index] = R_dest

    P = np.stack((P_left, P_up, P_down,P_right))
    R = np.stack((R_left, R_up, R_down,R_right))

    MDP = [P, R]

    return MDP


# by default, actions are ordered
# [left, up, down, right]
def get_optimal_policy(k_states = 38):
    """
    The functions return a 2D arrays: (38 x 4), corresponding to the (probabilistic) optimal policy for each state.
    By default, actions are ordered [left, up, down, right]
    """
    n_actions = 4

    a_left = 0
    a_up = 1
    a_down = 2
    a_right = 3

    opt_policy = np.zeros((k_states,n_actions))

    for state_src_index in range(k_states):
        state_src_label = state_src_index + 1

        if state_src_label == 2: # goal
            policy = a_left # any
            policy_prob = 1
        elif state_src_label == 1: # only go up
            policy = a_up
            policy_prob = 1
        elif state_src_label >= k_states-2: # right column -> only go down
            policy = a_down
            policy_prob = 1
        elif (state_src_label+1) % 3 == 0: # bottom row -> only go right
            policy = a_right
            policy_prob = 1
        else: # either right or down (only right)
            policy = a_right
            policy_prob = 1

        opt_policy[state_src_index,policy] = policy_prob

    # safety check
    opt_policy = normalize1_row(opt_policy)

    return opt_policy


def get_pi_e_greedy(pi, epsilon):
    """
    The functions return a 2D arrays: (38 x 4), corresponding to the (probabilistic) optimal policy for each state.
    By default, actions are ordered [left, up, down, right]
    Original optimal policy 'pi' is recalculated, and the optimal action is set to a prob of 1-epsilon,
    while the remaining epsilon is split equally onto the remaining possible actions
    """

    pi_e_greedy = np.copy(pi)

    for index, state_actions in enumerate(pi):
        opt_col = np.where(state_actions==1)[0]

        if (len(opt_col) != 1):
            print("ERROR! More than 1 optimal action")
            break

        non_opt_cols = np.where(state_actions==0)[0]

        pi_e_greedy[index, opt_col] -= epsilon
        pi_e_greedy[index, non_opt_cols] = epsilon/len(non_opt_cols)

    # safety check
    pi_e_greedy = normalize1_row(pi_e_greedy)

    return pi_e_greedy


def get_P_R(MDP, pi):
    """
    The functions return two elements: 
    - a matrix (38 x 38), corresponding to the probability of going from each state to each state, under policy PI;
    - a vector (38,1), correspoding to the expected reward for each state
    """

    MDP_P = MDP[0]
    MDP_R = MDP[1]

    k_states = MDP_P.shape[1] # 38
    n_actions = MDP_P.shape[0] # 4

    P = np.zeros((k_states,k_states))
    R = np.zeros((k_states,k_states))

    for state_src_index, row in enumerate(P):
        P_state_src = np.zeros((n_actions,k_states))
        R_state_src = np.zeros((n_actions,k_states))

        #
        for action_i in range(n_actions):
            P_state_src[action_i,:] = MDP_P[action_i,state_src_index]
            R_state_src[action_i,:] = MDP_R[action_i,state_src_index]

        action_i = np.reshape(pi[state_src_index],(1,-1)) # action in row form
        
        P[state_src_index,:] = action_i@P_state_src
        R[state_src_index,:] = action_i@(P_state_src*R_state_src)
        
        # state_dest = np.nonzero(P[state_src_index])[0]
        # reward_dest = R[state_src_index,state_dest]
        # print("{} -> {} = {}".format(str(state_src_index+1), str(state_dest+1), str(reward_dest)))
    
    Re_s = R @ np.ones(k_states) # state k value is sum of k-th row

    return P, Re_s


def get_v(P, R, gamma):
    k_states = P.shape[0] # 38

    I = np.identity(k_states)

    v = np.linalg.inv(I - gamma*P)@R

    return v


# Generate a full trajectory under the policy pi.
# The trajectory must start in init_state and end in terminal
def gen_episode(MDP, pi, init_state, terminal, gamma = 0.9):

    episode = [init_state]

    discounted_reward = 0

    MDP_P = MDP[0]
    MDP_R = MDP[1]

    current_state = init_state

    count_iterations = 0

    while current_state != terminal:
        count_iterations += 1

        pi_current_state = pi[current_state,:]
        
        randomize_action = np.random.rand()

        action = get_value(pi_current_state, randomize_action)

        randomize_destination = np.random.rand()

        prob_current_state = MDP_P[action,current_state]

        next_state = get_value(prob_current_state, randomize_destination)

        reward = MDP_R[action,current_state,next_state]

        episode.append(next_state)

        discounted_reward += reward*(gamma**(count_iterations-1))

        current_state = next_state

    return discounted_reward, count_iterations


def get_value(pi, value):
    pi_cumsum = np.cumsum(pi)

    action = np.where(pi_cumsum >= value)[0][0]

    return action


def normalize1_row(data):

    data_norm = data

    for i, row in enumerate(data):
        data_norm[i] = data[i]/np.sum(data[i])

    return data_norm


def get_avg_std(data_list):
    data_array = np.asfarray(data_list)

    data_avg = round(np.mean(data_array),2)
    data_std = round(np.std(data_array),2)

    return data_avg, data_std


if __name__ == "__main__":

    random.seed(1)

    k_states = 38 # for testing, 14 and 11 were used. Easier to see and debug

    gamma = 0.9 # discount factor
    init_state = 0 # \in [0,k_states)

    cliff_mdp = get_cliff_mdp(k_states)

    pi = get_optimal_policy(k_states)

    # YOUR CODE GOES HERE

    epsilon_values = [0.0, 0.1, 0.2]

    n_episodes = 1000
    print("Episodes: {}".format(n_episodes))

    for i, epsilon in enumerate(epsilon_values):
        print("## Epsilon: {} ##".format(epsilon))

        pi_e_greedy = get_pi_e_greedy(pi, epsilon)

        [P, Re_s] = get_P_R(cliff_mdp, pi_e_greedy)

        v = get_v(P, Re_s, gamma)

        print("State values:\n{}".format(v.round(2)))
        print("Initial state (label {}) value: {}".format(init_state+1, v[init_state].round(2)))

        transitions_list = []
        discounted_reward_avg = 0
        for epi in range(n_episodes):
            discounted_reward_i, transitions_i = gen_episode(cliff_mdp, pi_e_greedy, init_state, 1, gamma)

            discounted_reward_avg += discounted_reward_i
            transitions_list.append(transitions_i)

        discounted_reward_avg /= n_episodes

        transitions_avg, transitions_std = get_avg_std(transitions_list)

        print("Discounted reward average: {}".format(round(discounted_reward_avg,2)))
        print("Transitions average on episodes: {} +- {}".format(transitions_avg, transitions_std))
