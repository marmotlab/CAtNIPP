import copy
import os

import imageio
import numpy as np
import torch
from env import Env
from attention_net import AttentionNet
from parameters import *
import scipy.signal as signal

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False):

        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.sample_size = sample_size

        self.env = Env(sample_size=self.sample_size, k_size=K_SIZE, budget_range=budget_range, save_image=self.save_image)
        # self.local_net = AttentionNet(2, 128, device=self.device)
        # self.local_net.to(device)
        self.local_net = localNetwork
        self.experience = None

    def run_episode(self, currEpisode):
        episode_buffer = []
        perf_metrics = dict()
        for i in range(13):
            episode_buffer.append([])

        done = False
        node_coords, graph, node_info, node_std, budget = self.env.reset()
        
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

        mask = torch.zeros((1, self.sample_size+2, K_SIZE), dtype=torch.int64).to(self.device)
        for i in range(256):
            episode_buffer[9] += LSTM_h
            episode_buffer[10] += LSTM_c
            episode_buffer[11] += mask
            episode_buffer[12] += pos_encoding

            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            # next_node (1), logp_list (1, 10), value (1,1,1)
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer[4] += value
            episode_buffer[8] += budget_inputs 

            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())
            reward, done, node_info, node_std, remain_budget = self.env.step(next_node_index.item(), self.sample_length)
            #if (not done and i==127):
                #reward += -np.linalg.norm(self.env.node_coords[self.env.current_node_index,:]-self.env.node_coords[0,:])

            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)
       

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            #print(node_inputs)
            
            # mask last five node
            mask = torch.zeros((1, self.sample_size+2, K_SIZE), dtype=torch.int64).to(self.device)
            #connected_nodes = edge_inputs[0, current_index.item()]
            #current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, K_SIZE))
            #current_edge = current_edge.permute(0, 2, 1)
            #connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge) # (1, k_size, 1)
            #n_available_node = sum(int(x>0) for x in connected_nodes_budget.squeeze(0))
            #if n_available_node > 5:
            #    for j, node in enumerate(connected_nodes.squeeze(0)):
            #        if node.item() in route[-2:]:
            #            mask[0, route[-1], j] = 1

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot(route, self.global_step, i, gifs_path)

            if done:
                episode_buffer[6] = episode_buffer[4][1:]
                episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                if self.env.current_node_index == 0:
                    perf_metrics['remain_budget'] = remain_budget / budget
                    #perf_metrics['collect_info'] = 1 - remain_info.sum()
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = True
                    print('{} Goodbye world! We did it!'.format(i))
                else:
                    perf_metrics['remain_budget'] = np.nan
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_MI(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = False
                    print('{} Overbudget!'.format(i))
                break
        if not done:
            episode_buffer[6] = episode_buffer[4][1:]
            with torch.no_grad():
                 _, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            episode_buffer[6].append(value.squeeze(0))
            perf_metrics['remain_budget'] = remain_budget / budget
            perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
            perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
            perf_metrics['delta_cov_trace'] =  self.env.cov_trace0 - self.env.cov_trace
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
            perf_metrics['cov_trace'] = self.env.cov_trace
            perf_metrics['success_rate'] = False

        print('route is ', route)
        reward = copy.deepcopy(episode_buffer[5])
        reward.append(episode_buffer[6][-1])
        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward,dtype=object).reshape(-1)
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)

        for i in range(target_v.size()[0]):
            episode_buffer[7].append(target_v[i,:,:])

        # save gif
        if self.save_image:
            if self.greedy:
                from test_driver import result_path as path
            else:
                path = gifs_path
            self.make_gif(path, currEpisode)

        self.experience = episode_buffer
        return perf_metrics

    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

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
        A_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        D_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        for i in range(self.sample_size+2):
            for j in range(self.sample_size+2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size+2):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        L = np.eye(self.sample_size+2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:,1:32+1]
        return eigen_vector
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)




if __name__=='__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)
