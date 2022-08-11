from envs.Data_generation import VRPDataset
import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
from scipy.spatial import distance_matrix


class MultiAgentEnv(object):
    def __init__(self, filename, size, num_samples, offset, seed_value, distribution, index, seed):
        data = VRPDataset(filename, size, num_samples, offset, seed_value, distribution)[index]
        self.UAV_num = data['UAV_num']
        self.loc = data['loc']
        self.data_size = len(data['alpha'])
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.base = data['base']
        self.velocity = data['velocity']
        self.capacity = data['capacity']
        self.depot = data['depot']
        self.state = np.concatenate((self.loc, self.alpha[:, np.newaxis], self.beta[:, np.newaxis], self.base[:, np.newaxis]), axis=1)
        self.initial_state = np.concatenate((self.depot, np.zeros(3)))
        self.route_state = []
        self.route = []
        self.route_time = []
        self.reward_record = []
        self.temp_reward_record = []
        for agent in range(self.UAV_num):
            self.route_state.append([self.initial_state])
            self.route.append([0])
            self.route_time.append([0])
            self.temp_reward_record.append([0])
        self.dis_matrix = distance_matrix(np.concatenate((np.expand_dims(self.depot, axis=0), self.loc), axis=0), np.concatenate((np.expand_dims(self.depot, axis=0), self.loc), axis=0))
        self.travel_time = self.dis_matrix / self.velocity
        self.complete_node = []
        self.episode_limit = 100

    def step(self, actions):
        """ Returns reward, terminated, info """
        action = actions.view(self.UAV_num, -1)
        terminated = np.zeros([self.UAV_num])
        terminate = 1
        reward = 0
        for agent in range(self.UAV_num):
            step_action = int(action[agent])
            if step_action in self.complete_node:
                continue
            if step_action == 0:
                terminated[agent] = 1
            if step_action == 0 and self.route[agent][-1] == 0 and len(self.route[agent]) != 1:
                continue
            self.route[agent].append(step_action)
            self.route_state[agent].append(self.state[step_action - 1])
            if step_action != 0:
                self.complete_node.append(step_action)
            if self.route[agent][-2] == 0:
                self.route_time[agent].append(self.travel_time[self.route[agent][0]][self.route[agent][1]])
            elif self.route[agent][-1] == 0:
                t = self.capacity - self.travel_time[self.route[agent][-2]][self.route[agent][-1]] - self.route_time[agent][-1]
                service_start_time = self.route_time[agent][-1]
                service_end_time = self.route_time[agent][-1] + t
                a = self.base[self.route[agent][-2] - 1]
                b = self.value_decay(service_start_time, self.alpha[self.route[agent][-2] - 1], self.beta[self.route[agent][-2] - 1])
                temp_reward = self.information_value_cum(self.information_value, service_start_time, service_end_time, a, b)
                self.route_time[agent].append(self.capacity)
                self.temp_reward_record[agent].append(temp_reward[0])
                self.temp_reward_record[agent].append(0)
            else:
                node1 = self.route[agent][-2]
                node2 = self.route[agent][-1]
                func1 = self.value_decay
                func2 = self.information_value
                func3 = self.information_value_cum
                a1 = self.base[node1 - 1]
                a2 = self.base[node2 - 1]
                alpha1 = self.alpha[node1 - 1]
                alpha2 = self.alpha[node2 - 1]
                beta1 = self.beta[node1 - 1]
                beta2 = self.beta[node2 - 1]
                start_time = self.route_time[agent][-1]
                left_time = self.capacity - start_time - self.travel_time[node1][node2] - self.travel_time[node2][0]
                travel_time = self.travel_time[node1][node2]
                temp_t, temp_reward = self.calculate_t(func1, func2, func3, a1, a2, alpha1, alpha2, beta1, beta2, start_time, left_time, travel_time)
                self.route_time[agent].append(temp_t + self.route_time[agent][-1] + travel_time)
                self.temp_reward_record[agent].append(temp_reward[0])
        for agent in range(self.UAV_num):
            terminate *= terminated[agent]
        if terminate == 1:
            for agent in range(self.UAV_num):
                for i in range(len(self.temp_reward_record[agent])):
                    reward += self.temp_reward_record[agent][i]
            self.reward_record.append(reward)
        return reward, terminate, self.get_env_info()

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_mask = np.ones([self.UAV_num, len(self.state), len(self.state[0])])
        for item in self.complete_node:
            obs_mask[:, item-1, :] = 0
        return (obs_mask * np.expand_dims(self.state, axis=0).repeat([self.UAV_num], axis=0)).reshape(1, self.UAV_num, len(self.state[0]) * len(self.state))

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.state[0]) * len(self.state)

    def get_state(self):
        new_state = self.route_state[0][-1]
        for agent in range(1, self.UAV_num):
            new_state = np.concatenate((new_state, self.route_state[agent][-1]), axis=0)
        return new_state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.UAV_num * len(self.state[0])

    def get_avail_actions(self):
        avail_actions = np.ones([self.UAV_num, self.get_total_actions()])
        for agent in range(self.UAV_num):
            temp_node = self.route[agent][-1]
            temp_node_time = self.route_time[agent][-1]
            if temp_node == 0 and len(self.route[agent]) != 1:
                for i in range(1, self.get_total_actions()):
                    avail_actions[agent][i] = 0
            for item in self.complete_node:
                avail_actions[agent][item] = 0
            for node in range(1, 1 + len(self.loc)):
                if avail_actions[agent][node] != 0:
                    avail_time = self.capacity - temp_node_time - self.travel_time[node][temp_node] - self.travel_time[node][0]
                    if avail_time < 0:
                        avail_actions[agent][node] = 0
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.data_size + 1

    def reset(self):
        """ Returns initial observations and states"""
        self.route_state = []
        self.route = []
        self.route_time = []
        self.temp_reward_record = []
        for agent in range(self.UAV_num):
            self.route_state.append([self.initial_state])
            self.route.append([0])
            self.route_time.append([0])
            self.temp_reward_record.append([0])
        self.complete_node = []

    def render(self):
        raise NotImplementedError

    def close(self):
        print("route_state", self.route_state)
        print("route", self.route)
        print("route_time", self.route_time)
        print("reward", self.reward_record[-100:])
        total_loc = np.concatenate((np.expand_dims(self.depot, axis=0), self.loc), axis=0)
        return total_loc, self.route, self.route_time, self.reward_record

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.UAV_num,
                    "episode_limit": self.episode_limit}
        return env_info

    def value_decay(self, x, alpha, beta):
        return beta / pow((np.exp(x) + 1), alpha)

    def information_value(self, x, a, b):
        return ((np.exp(-x + a) - np.exp(x - a)) / (np.exp(x - a) + np.exp(-x + a)) + 1) * b / 2

    def information_value_cum(self, func, service_start_time, service_end_time, a, b):
        return integrate.quad(func, service_start_time, service_end_time, args=(a, b))

    def func_T(self, T, func, a, b):
        return [func(T, a, b)[0]]

    def func_t(self, t, func1, func2, func3, a1, a2, b1, b2, alpha2, beta2, start_time, T1, T2, travel_time):
        b3 = func1(start_time + t + travel_time, alpha2, beta2)
        T3 = fsolve(self.func_T, a2, args=(func2, a2, b3))[0]
        return [func3(func2, t, T1, a1, b1)[0] + func3(func2, 0, T2, a2, b2)[0] - func3(func2, 0, T3, a2, b3)[0]]

    def calculate_final_t(self, func1, func2, func3, alpha, beta, a, start_time, left_time):
        b = func1(start_time, alpha, beta)
        T = fsolve(self.func_T, a, args=(func2, a, b))[0]
        T = min(T, left_time)
        value = func3(func2, 0, T, a, b)
        return T, value

    def calculate_t(self, func1, func2, func3, a1, a2, alpha1, alpha2, beta1, beta2, start_time, left_time, travel_time):
        b1 = func1(start_time, alpha1, beta1)
        T1 = fsolve(self.func_T, a1, args=(func2, a1, b1))[0]
        ##print("T1", T1)
        b2 = func1(start_time + T1 + travel_time, alpha2, beta2)
        T2 = fsolve(self.func_T, a2, args=(func2, a2, b2))[0]
        ##print("T2", T2)
        T2 = min(T2, left_time)
        # judge if condition exists
        b3 = func1(start_time + travel_time, alpha2, beta2)
        T3 = fsolve(self.func_T, a2, args=(func2, a2, b3))[0]
        if func3(func2, 0, T1, a1, b1)[0] + func3(func2, 0, T2, a2, b2)[0] - func3(func2, 0, T3, a2, b3)[0] < 0:
            return 0, [0]
        t = \
        fsolve(self.func_t, 0, args=(func1, func2, func3, a1, a2, b1, b2, alpha2, beta2, start_time, T1, T2, travel_time))[0]
        ##print(ans)
        Value = func3(func2, 0, t, a1, b1)
        return t, Value

"""
if __name__ == "__main__":
    data = VRPDataset()
    a = MultiAgentEnv(data[9])
    print(a.loc)
"""