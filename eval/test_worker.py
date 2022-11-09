import copy
import os
import imageio
import numpy as np
import time
import ray
import torch
from env import Env
from attention_net import AttentionNet
import scipy.signal as signal
from multiprocessing import Pool
from test_parameters import *


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class WorkerTest:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_length=None, device='cuda', greedy=False, save_image=False, seed=None):

        self.device = device
        self.greedy = True
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.seed = seed
        self.env = Env(sample_size=SAMPLE_SIZE, start=(0,0), destination=(1,1), k_size=K_SIZE, budget_range=budget_range, save_image=self.save_image, seed=seed)

        self.local_net = localNetwork
        self.perf_metrics = None
        self.budget_history =[]
        self.obj_history = []
        self.obj2_history = []
        self.planning_time = 0

    def run_episode(self, currEpisode, testID):
        perf_metrics = dict()

        done = False
        node_coords, graph, node_info, node_std, budget = self.env.reset()
        np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        n_nodes = node_coords.shape[0]
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes,1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1)
        node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 4)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 1)
        
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device) # (1, sample_size+2, 32)

        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, k_size)

        current_index = torch.tensor([self.env.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,1)
        route = [current_index.item()]

        LSTM_h = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)

        for i in range(256):
            #if len(route) >= 2:
            #    mask = torch.zeros((1, SAMPLE_SIZE+2, K_SIZE), dtype=torch.int64).to(self.device)
            #    connected_nodes = edge_inputs[0, current_index.item()]
            #    for j, node in enumerate(connected_nodes.squeeze(0)):
            #        if node.item() in route[-5:]:
            #            mask[0, route[-1], j] = 1
            #else:
            #    mask = None

            t1 = time.time()
            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding)
            # next_node (1), logp_list (1, 10), value (1,1,1)
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            t2 = time.time()
            self.planning_time += t2-t1

            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())
            reward, done, node_info, node_std, remain_budget = self.env.step(next_node_index.item(), self.sample_length)
            self.budget_history.append(budget-remain_budget)
            self.obj_history.append(self.env.cov_trace)
            self.obj2_history.append(self.env.RMSE)

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            #print(node_inputs)

            # save a frame
            if self.save_image:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                self.env.plot(route, self.global_step, i, result_path, testID)

            if done:
                if self.env.current_node_index == 0:
                    perf_metrics['budget'] = budget
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = True
                    perf_metrics['budget_history'] = self.budget_history
                    perf_metrics['obj_history'] = self.obj_history
                    perf_metrics['obj2_history'] = self.obj2_history
                    perf_metrics['planning_time'] = self.planning_time
                    print('{} Goodbye world! We did it!'.format(i))
                break

        print('route is ', route)
        # save gif
        if self.save_image:
            self.make_gif(result_path, currEpisode, testID)
        return perf_metrics

    def work(self, currEpisode, testID):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        print("starting testing episode {} test {} on metaAgent {}".format(currEpisode, testID, self.metaAgentID))
        self.currEpisode = currEpisode
        if TRAJECTORY_SAMPLING:
            self.perf_metrics = self.run_trajectory_sampling_episode(currEpisode, testID)
        else:
            self.perf_metrics = self.run_episode(currEpisode, testID)
        return self.perf_metrics

    def run_trajectory_sampling_episode(self, currEpisode, testID):
        perf_metrics = dict()

        done = False
        node_coords, graph, node_info, node_std, budget = self.env.reset()
        np.random.seed(self.seed)
        #torch.manual_seed(self.seed)
        n_nodes = node_coords.shape[0]
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes, 1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1)
        #node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
        #node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 4)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 1)

        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)
        
        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device) # (1, sample_size+2, 32)
        
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, k_size)

        current_index = torch.tensor([self.env.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        route = [current_index.item()]

        LSTM_h = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)

        for i in range(256):
            t1 = time.time()
            temp_env = copy.deepcopy(self.env)

            p = Pool(processes=NUM_SAMPLE_TEST)
            results = []
            results.append(p.apply_async(self.plan_route, args=(-1, temp_env, node_coords, node_info_inputs, node_std_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, True)))
            
            for j in range(NUM_SAMPLE_TEST-1):
                results.append(p.apply_async(self.plan_route, args=(j, temp_env, node_coords, node_info_inputs, node_std_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding)))
            p.close()
            p.join()

            best = np.inf
            best_route = None
            for res in results:
                cov_trace, temp_route, candi_LSTM_h, candi_LSTM_c = res.get()
                #print(temp_route)
                if cov_trace < best:
                    best = cov_trace
                    best_route = temp_route
                    LSTM_h, LSTM_c = candi_LSTM_h, candi_LSTM_c

            #_, best_route, LSTM_h, LSTM_c = self.plan_route(temp_env, node_coords, node_info_inputs, node_std_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c)
            
            t2 = time.time()
            self.planning_time += t2-t1

            k=0
            for next_node_index in best_route[:3]:
                route.append(next_node_index)
                _, done, node_info, node_std, remain_budget = self.env.step(next_node_index, self.sample_length)
                k += 1
            self.budget_history.append(budget - remain_budget)
            self.obj_history.append(self.env.cov_trace)
            self.obj2_history.append(self.env.RMSE)

            if self.save_image:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                self.env.plot(route, self.global_step, i*PLAN_STEP+k, result_path, testID, sampling_path=best_route)

            current_index = torch.tensor([next_node_index], dtype=torch.int64).unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            #node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            #node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)

            if done:
                if self.env.current_node_index == 0:
                    perf_metrics['budget'] = budget
                    # perf_metrics['collect_info'] = 1 - remain_info.sum()
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = True
                    perf_metrics['budget_history'] = self.budget_history
                    perf_metrics['obj_history'] = self.obj_history
                    perf_metrics['obj2_history'] = self.obj2_history
                    perf_metrics['planning_time'] = self.planning_time
                    print('{} Goodbye world! We did it!'.format(i))
                break

        print('route is ', route)
        # save gif
        if self.save_image:
            self.make_gif(result_path, currEpisode, testID)
        return perf_metrics

    def plan_route(self, num, env, node_coords, node_info_inputs, node_std_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, greedy=False):
        temp_route = []
        node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        for j in range(PLAN_STEP):
            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs,
                                                                  current_index, LSTM_h, LSTM_c, pos_encoding)
            torch.manual_seed(int(time.time())+100*num)
            if greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)
            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            temp_route.append(next_node_index.item())

            _, done, _, node_std, remain_budget = env.step(next_node_index.item(), self.sample_length, measurement=False)
            if done:
                break

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_std_inputs = node_std.reshape(-1, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
        return env.cov_trace, temp_route, LSTM_h, LSTM_c

    def calc_estimate_budget(self, budget, current_idx):
        all_budget = []
        current_coord = self.env.node_coords[current_idx]
        end_coord = self.env.node_coords[0]
        for i, point_coord in enumerate(self.env.node_coords):
            dist_current2point = self.env.prm.calcDistance(current_coord, point_coord)
            dist_point2end = self.env.prm.calcDistance(point_coord, end_coord)
            estimate_budget = (budget - dist_current2point - dist_point2end) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i+1, 1)

    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((SAMPLE_SIZE+2, SAMPLE_SIZE+2))
        D_matrix = np.zeros((SAMPLE_SIZE+2, SAMPLE_SIZE+2))
        for i in range(SAMPLE_SIZE+2):
            for j in range(SAMPLE_SIZE+2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(SAMPLE_SIZE+2):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        L = np.eye(SAMPLE_SIZE+2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:,1:32+1]
        return eigen_vector 
    
    def make_gif(self, path, n, testID):
        with imageio.get_writer('{}/ep{}_test{}_cov_trace_{:.4g}.gif'.format(path, n, testID, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files:
            os.remove(filename)




if __name__=='__main__':
    pass
