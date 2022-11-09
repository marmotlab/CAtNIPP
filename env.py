import os
import copy
import numpy as np
from itertools import product
from classes import PRMController, Obstacle, Utils
from classes.Gaussian2D import Gaussian2D
from matplotlib import pyplot as plt
from gp_ipp import GaussianProcessForIPP
from parameters import ADAPTIVE_AREA



class Env():
    def __init__(self, sample_size=500, k_size=10, start=None, destination=None, obstacle=[], budget_range=None, save_image=False, seed=None):
        self.sample_size = sample_size
        self.k_size = k_size
        self.budget_range = budget_range
        self.budget = np.random.uniform(*self.budget_range)
        if start is None:
            self.start = np.random.rand(1, 2)
        else:
            self.start = np.array([start])
        if destination is None:
            self.destination = np.random.rand(1, 2)
        else:
            self.destination = np.array([destination])
        self.obstacle = obstacle
        self.seed = seed
        
        # generate PRM
        # self.prm = None
        # self.node_coords, self.graph = None, None
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)
        self.prm = PRMController(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range,
                                 self.k_size)
        self.budget = np.random.uniform(*self.budget_range)
        self.node_coords, self.graph = self.prm.runPRM(saveImage=False, seed=seed)
        
        # underlying distribution
        self.underlying_distribution = None
        self.ground_truth = None
        self.high_info_area = None

        # GP
        self.gp_ipp = None
        self.node_info, self.node_std = None, None
        self.node_info0, self.node_std0, self.budget0 = copy.deepcopy((self.node_info, self.node_std,self.budget))
        self.RMSE = None
        self.F1score = None
        self.cov_trace = None
        self.MI = None
        self.MI0 = None
        
        # start point
        self.current_node_index = 1
        self.sample = self.start
        self.dist_residual = 0
        self.route = []

        self.save_image = save_image
        self.frame_files = []

    def reset(self, seed=None):
        # generate PRM
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)
        # self.prm = PRMController(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range, self.k_size)
        # self.budget = np.random.uniform(*self.budget_range)
        # self.node_coords, self.graph = self.prm.runPRM(saveImage=False)
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        # underlying distribution
        self.underlying_distribution = Gaussian2D()
        self.ground_truth = self.get_ground_truth()

        # initialize gp
        self.gp_ipp = GaussianProcessForIPP(self.node_coords)
        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        self.node_info, self.node_std = self.gp_ipp.update_node()
        
        # initialize evaluations
        #self.F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
        self.RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        self.cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        self.MI = self.gp_ipp.evaluate_mutual_info(self.high_info_area)
        self.cov_trace0 = self.cov_trace

        # save initial state
        self.node_info0, self.node_std0, self.budget = copy.deepcopy((self.node_info, self.node_std,self.budget0))

        # start point
        self.current_node_index = 1
        self.sample = self.start
        self.dist_residual = 0
        self.route = []
        np.random.seed(None)

        return self.node_coords, self.graph, self.node_info, self.node_std, self.budget

    def step(self, next_node_index, sample_length, measurement=True):
        dist = np.linalg.norm(self.node_coords[self.current_node_index] - self.node_coords[next_node_index])
        remain_length = dist
        next_length = sample_length - self.dist_residual
        reward = 0
        done = True if next_node_index == 0 else False
        no_sample = True
        while remain_length > next_length:
            if no_sample:
                self.sample = (self.node_coords[next_node_index] - self.node_coords[
                    self.current_node_index]) * next_length / dist + self.node_coords[self.current_node_index]
            else:
                self.sample = (self.node_coords[next_node_index] - self.node_coords[
                    self.current_node_index]) * next_length / dist + self.sample
            if measurement:
                observed_value = self.underlying_distribution.distribution_function(
                    self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
            else:
                observed_value = np.array([0])
            self.gp_ipp.add_observed_point(self.sample, observed_value)

            remain_length -= next_length
            next_length = sample_length
            no_sample = False

        self.gp_ipp.update_gp()
        self.node_info, self.node_std = self.gp_ipp.update_node()

        if measurement:
            self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        #F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
            RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
            self.RMSE = RMSE
        cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        #self.F1score = F1score
        if next_node_index in self.route[-2:]:
            reward += -0.1

        elif self.cov_trace > cov_trace:
            reward += (self.cov_trace - cov_trace) / self.cov_trace
        self.cov_trace = cov_trace

        if done:
            reward -= cov_trace/900

        self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
        self.budget -= dist
        self.current_node_index = next_node_index
        self.route.append(next_node_index)
        assert self.budget >= 0  # Dijsktra filter
         
        return reward, done, self.node_info, self.node_std, self.budget

    def route_step(self, route, sample_length, measurement=True):
        current_node = route[0]
        for next_node in route[1:]:
            dist = np.linalg.norm(current_node - next_node)
            remain_length = dist
            next_length = sample_length - self.dist_residual
            no_sample = True
            while remain_length > next_length:
                if no_sample:
                    self.sample = (next_node - current_node) * next_length / dist + current_node
                else:
                    self.sample = (next_node - current_node) * next_length / dist + self.sample
                observed_value = self.underlying_distribution.distribution_function(self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                self.gp_ipp.add_observed_point(self.sample, observed_value)
                remain_length -= next_length
                next_length = sample_length
                no_sample = False

            self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
            self.dist_residual_tmp = self.dist_residual
            if measurement:
                self.budget -= dist
            current_node = next_node

        self.gp_ipp.update_gp()

        if measurement:
            self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
            cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
            self.cov_trace = cov_trace
        else:
            cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)

        return cov_trace

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth

    def plot(self, route, n, step, path, testID=0, CMAES_route=False, sampling_path=False):
        # Plotting shorest path
        plt.switch_backend('agg')
        self.gp_ipp.plot(self.ground_truth)
        # plt.subplot(1,3,1)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)
        if CMAES_route:
            pointsToDisplay = route
        else:
            pointsToDisplay = [(self.prm.findPointsFromNode(path)) for path in route]
        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]
        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        if sampling_path:
            pointsToDisplay2 = [(self.prm.findPointsFromNode(path)) for path in sampling_path]
            x0 = [item[0] for item in pointsToDisplay2]
            y0 = [item[1] for item in pointsToDisplay2]
            x1 = [item[0] for item in pointsToDisplay2[:3]]
            y1 = [item[1] for item in pointsToDisplay2[:3]]
            for i in range(len(x0) - 1):
                plt.plot(x0[i:i + 2], y0[i:i + 2], c='white', linewidth=4, zorder=5, alpha=1- 0.2 * i / len(x0))
            for i in range(len(x1) - 1):
                plt.plot(x1[i:i + 2], y1[i:i + 2], c='red', linewidth=4, zorder=6)

        plt.subplot(2,2,4)
        plt.title('High interest area')
        xh = self.high_info_area[:,0]
        yh = self.high_info_area[:,1]
        plt.hist2d(xh, yh, bins=30, range=[[0,1], [0,1]], vmin=0, vmax=1, rasterized=True)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)

        # x = [item[0] for item in pointsToDisplay]
        # y = [item[1] for item in pointsToDisplay]

        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        plt.suptitle('Budget: {:.4g}/{:.4g},  Cov trace: {:.4g}'.format(
            self.budget, self.budget0, self.cov_trace))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size), dpi=150)
        # plt.show()
        frame = '{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size)
        self.frame_files.append(frame)

if __name__=='__main__':
    env = Env(sample_size=200, budget_range=(7.999,8), save_image=True)
    nodes, graph, info, std, budget = env.reset()
    print(nodes)
    print(graph)

