import copy
from collections import defaultdict
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shapely.geometry
import argparse

from .Graph import Graph, dijkstra, to_array
from .Utils import Utils


class PRMController:
    def __init__(self, sample_size, allObs, start, destination, budget_range, k_size):
        self.sample_size = sample_size
        self.node_coords = np.array([])
        self.allObs = allObs
        self.start = np.array(start)
        self.destination = np.array(destination)
        self.graph = Graph()
        self.utils = Utils()
        self.k_size = k_size
        self.dijkstra_dist = []
        self.dijkstra_prev = []

    def runPRM(self, seed=None, saveImage=False):
        if saveImage:
            utils = Utils()
            utils.drawMap(self.allObs, self.start, self.destination)
        if seed is not None:
            np.random.seed(seed)

        # Generate n random samples called milestones
        self.genCoords()

        # Check if milestones are collision free
        self.checkIfCollisonFree()

        # Link each milestone to k nearest neighbours.
        # Retain collision free links as local paths.
        self.findNearestNeighbour(k=self.k_size, save_image=saveImage)
        self.calcAllPathCost()
        # Search for shortest path from start to end node - Using Dijksta's shortest path alg
        # self.shortestPath()

        # if saveImage:
        #     self.plotPoints(self.collisionFreePoints)

        return self.collisionFreePoints, self.graph.edges

    def genCoords(self):
        self.node_coords = np.random.rand(self.sample_size, 2)
        # from itertools import product
        # self.node_coords = np.array(list(product(np.linspace(0.05,0.95,7), np.linspace(0.05,0.95,7))))
        # Adding begin and end points
        self.start = self.start.reshape(1, 2)
        self.destination = self.destination.reshape(1, 2)
        self.node_coords = np.concatenate(
            (self.destination, self.start, self.node_coords), axis=0)

    def checkIfCollisonFree(self):
        collision = False
        self.collisionFreePoints = np.array([])
        for point in self.node_coords:
            collision = self.checkPointCollision(point)
            if(not collision):
                if(self.collisionFreePoints.size == 0):
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point])


    def findNearestNeighbour(self, k, save_image):
        X = self.collisionFreePoints
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 2), int)

        for i, p in enumerate(X):
            # Ignoring nearest neighbour - nearest neighbour is the point itself
            for j, neighbour in enumerate(X[indices[i][:]]):
                start_line = p
                end_line = neighbour
                if(not self.checkPointCollision(start_line) and not self.checkPointCollision(end_line)):
                    if(not self.checkLineCollision(start_line, end_line)):
                        self.collisionFreePaths = np.concatenate(
                            (self.collisionFreePaths, p.reshape(1, 2), neighbour.reshape(1, 2)), axis=0)

                        a = str(self.findNodeIndex(p))
                        b = str(self.findNodeIndex(neighbour))
                        self.graph.add_node(a)
                        self.graph.add_edge(a, b, distances[i, j])
                        # if save_image:
                        #     x = [p[0], neighbour[0]]
                        #     y = [p[1], neighbour[1]]
                        #     plt.plot(x, y, c='tan', alpha=0.4)

    def calcAllPathCost(self):
        for coord in self.collisionFreePoints:
            startNode = str(self.findNodeIndex(coord))
            dist, prev = dijkstra(self.graph, startNode)
            self.dijkstra_dist.append(dist)
            self.dijkstra_prev.append(prev)

    def calcDistance(self, current, destination):
        startNode = str(self.findNodeIndex(current))
        endNode = str(self.findNodeIndex(destination))
        if startNode == endNode:
            return 0
        pathToEnd = to_array(self.dijkstra_prev[int(startNode)], endNode)
        if len(pathToEnd) <= 1: # not expand this node
            return 1000

        distance = self.dijkstra_dist[int(startNode)][endNode]
        distance = 0 if distance is None else distance
        return distance

    def shortestPath(self, current, destination):
        self.startNode = str(self.findNodeIndex(current))
        self.endNode = str(self.findNodeIndex(destination))
        if self.startNode == self.endNode:
            return 0
        dist, prev = dijkstra(self.graph, self.startNode)

        pathToEnd = to_array(prev, self.endNode)

        if len(pathToEnd) <= 1: # not expand this node
            return 1000

        # pointsToEnd = [str(self.findPointsFromNode(path)) for path in pathToEnd]
        distance = dist[self.endNode]
        # print("****Output****")
        #
        # print("The quickest path from {} to {} is: \n {} \n with a distance of {}".format(
        #     self.collisionFreePoints[int(self.startNode)],
        #     self.collisionFreePoints[int(self.endNode)],
        #     " \n ".join(pointsToEnd),
        #     str(distance)))
        distance = 0 if distance is None else distance
        return distance

    def checkLineCollision(self, start_line, end_line):
        collision = False
        line = shapely.geometry.LineString([start_line, end_line])
        for obs in self.allObs:
            if(self.utils.isWall(obs)):
                uniqueCords = np.unique(obs.allCords, axis=0)
                wall = shapely.geometry.LineString(
                    uniqueCords)
                if(line.intersection(wall)):
                    collision = True
            else:
                obstacleShape = shapely.geometry.Polygon(
                    obs.allCords)
                collision = line.intersects(obstacleShape)
            if(collision):
                return True
        return False

    def findNodeIndex(self, p):
        # return np.where((self.collisionFreePoints == p).all(axis=1))[0][0]
        # print(np.linalg.norm(self.collisionFreePoints - p, axis=1))
        return np.where(np.linalg.norm(self.collisionFreePoints - p, axis=1) < 1e-5)[0][0]

    def findPointsFromNode(self, n):
        return self.collisionFreePoints[int(n)]

    def plotPoints(self, points):
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        # plt.scatter(x, y, c=info, cmap='Blues', s=5, zorder=10)
        plt.scatter(x,y)
        plt.colorbar()

    def checkCollision(self, obs, point):
        p_x = point[0]
        p_y = point[1]
        if(obs.bottomLeft[0] <= p_x <= obs.bottomRight[0] and obs.bottomLeft[1] <= p_y <= obs.topLeft[1]):
            return True
        else:
            return False

    def checkPointCollision(self, point):
        for obs in self.allObs:
            collision = self.checkCollision(obs, point)
            if(collision):
                return True
        return False

