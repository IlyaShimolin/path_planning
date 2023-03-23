
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import heapq
import trimesh
from  kdquery import Tree
import pickle
import copy

from time import time

class Graph:
    def __init__(self, dim, capacity = 10000):
        self.edges = collections.defaultdict(list)
        self.kd = Tree(dim, capacity)
        self.start_id = None
        self.target_id = None
    def __len__(self):
        return len(self.kd)
    def add_node(self, joint):
        node_id = self.kd.insert(joint)
        return node_id
    def add_edge(self, joint1, joint2):
        self.edges[joint1].append(joint2)
        self.edges[joint2].append(joint1)
    def get_near_node_by_rad(self, joint, rad):
        return self.kd.find_points_within_radius(joint, rad)
    def get_point(self, node_id):
        return self.kd.get_node(node_id).point
    def get_edges(self, node_id):
        return self.edges[node_id]
    def get_nearest_point(self, joint):
        return  self.kd.find_nearest_point(joint)

class roadnode:
    def __init__(self):
        self.score = float('inf')
        self.parent = -1

class PRM:
    def __init__(self, robot,env,max_len_graph = 10000,r = 1.5):
        self.robot = robot
        
        self.env = env
        self.max_len_graph = max_len_graph
        self.max_near_node = 5
        self.rad = r
        self.joint_lim_h=np.array([180,0,160,360,360,360])*3.14/180
        self.joint_lim_l=np.array([-180,-180,-160,-360,-360,-360])*3.14/180
        self.names_link = []
        for link in robot.links:
            self.names_link.append(link.name)

        self.names_joints = []
        for joint in robot.joints:
            self.names_joints.append(joint.name)
        fk = self.robot.collision_trimesh_fk()
        self.arm_manager = trimesh.collision.CollisionManager()
        i=0
        for tm in fk:
            pose = fk[tm]
            self.arm_manager.add_object(name = self.names_link[i],mesh = tm, transform=pose)
            i += 1
        self.n_dof = len(self.joint_lim_h)

    def get_valid_joints_sample(self):
        joints = np.random.random(self.n_dof)*(self.joint_lim_h - self.joint_lim_l) + self.joint_lim_l
        return joints 
    
    def check_collision(self, joints):
        i = 0
        cfg = {self.names_joints[i]:joints[i] for i in range(self.n_dof)}
        fk = self.robot.collision_trimesh_fk(cfg=cfg)
        for tm in fk:
            pose = fk[tm]
            self.arm_manager.set_transform(name = self.names_link[i], transform=pose)
            i += 1

        cl = self.arm_manager.in_collision_internal(return_names = True)
        if cl[0] == True:
            result = True
            for i in cl[1]:
                result *= (i[1] == self.names_link[self.names_link.index(i[0])-1] or i[1] == self.names_link[self.names_link.index(i[0])+1])
            
        if self.arm_manager.in_collision_other(self.env):
            result = False
        return not result
    

    def check_pair_valid(self, joints_s,joints_s_n,step_size = 0.2):
        path = np.linspace(joints_s, joints_s_n, int(np.linalg.norm(joints_s_n-joints_s)/step_size) )
        for joint_in_path in path:
            if self.check_collision(joint_in_path):
                return False
        return True
    
    def generate_samples(self, graph):
        num_edges = 0
        while len(graph) < self.max_len_graph:
            joints_s = self.get_valid_joints_sample()
            if not self.check_collision(joints_s):
                node_id = graph.add_node(joints_s)
                near_node_ids = graph.get_near_node_by_rad(joints_s, self.rad)
                num_valid_near_node = 0
                for near_node_id in near_node_ids:
                    joints_s_n = graph.get_point(near_node_id)
                    if self.check_pair_valid(joints_s,joints_s_n):
                        graph.add_edge(node_id,near_node_id)
                        num_edges += 1
                        num_valid_near_node += 1
                        if num_valid_near_node >= self.max_near_node: 
                            break 
    #поиск пути по PRM
    def search(self, graph):

        open_list, closed_list = [], set()
        road_map = collections.defaultdict(roadnode)
        road_map[graph.start_id].score = 0
        heapq.heappush(open_list, (0, graph.start_id))


        path_unfound = True 

        while open_list and path_unfound:
            
             
            _, c_id = heapq.heappop(open_list)
            
            if c_id in closed_list:
                continue
            closed_list.add(c_id)
            near_ids = graph.get_edges(c_id)
            for next_id in near_ids:
                if next_id == graph.target_id:
                    path_unfound = False
                    road_map[graph.target_id].parent = c_id
                    break
                if road_map[next_id].score > road_map[c_id].score + np.linalg.norm(graph.get_point(next_id)-graph.get_point(c_id)):
                    road_map[next_id].score = road_map[c_id].score + np.linalg.norm(graph.get_point(next_id)-graph.get_point(c_id))
                    score = road_map[next_id].score
                    heapq.heappush(open_list,(score,next_id))
                    road_map[next_id].parent = c_id                                                  
        path = []
        print (path_unfound)
        if not path_unfound:
            backward_path = [graph.get_point(graph.target_id)]
            node_id = road_map[graph.target_id].parent
            while node_id != -1:
                backward_path.append(graph.get_point(node_id))
                node_id = (road_map[node_id]).parent
            path = backward_path[::-1]
        return path
    
    def get_pathlenght(self,path):
        if len(path) > 1:
            return 0
        path_to_print = path
        pth = np.linspace(path_to_print[0], path_to_print[1], 10)
        for i in range(len(path)-2):
            pth = np.concatenate((pth, np.linspace(path_to_print[i+1], path_to_print[i+2], 10)), axis=0)
        px,py,pz = [],[],[]
        for j in pth:
            x1,y1,z1=[],[],[]
            cfg = {self.names_joints[i]:j[i] for i in range(6)}
            fk = self.robot.collision_trimesh_fk(cfg=cfg) 
            for i in range(7):
                x1.append(list(fk.items())[i][1][0,3])
                y1.append(list(fk.items())[i][1][1,3])
                z1.append(list(fk.items())[i][1][2,3])
            px.append(x1[6])
            py.append(y1[6])
            pz.append(z1[6])
        #ax.scatter3D(x1,y1,z1, color = "blue")
        #ax.plot(x1,y1,z1, color = "blue")
        dist = 0
        for i in range(len(px)-1):
            dist += math.sqrt((px[i]-px[i+1])**2+(py[i]-py[i+1])**2+(pz[i]-pz[i+1])**2)
        return dist
    
    def get_dist(self, path_):
        dist = 0
        for i in range(len(path_)-1):
            dist += np.linalg.norm(path_[i+1]-path_[i])
        dist_ideal=0
        if len(path_)>0:
            dist_ideal = np.linalg.norm(path_[-1]-path_[0])
        return dist-dist_ideal
    #Оптимизация молученной траектории градиентным спуском   
    def get_grad(self, path_, i, j):

        SamplingDistance = 0.001
        f_x = self.get_dist(path_)
        path2_ = copy.deepcopy(path_)
        path2_[i][j] = path_[i][j] + SamplingDistance
        f_x_d = self.get_dist(path2_)
        grad = (f_x_d - f_x) / SamplingDistance
        return grad
    
    def path_valid_check(self, path_):
        result = True
        for i in range(len(path_)-1):
            result = result and self.check_pair_valid(path_[i],path_[i+1],step_size = 1)
        return result
    def grad_est(self,path, max_iter = 100, eps = 0.01):

        path_cur = copy.deepcopy(path)
        path_next = copy.deepcopy(path)

        L_rate = 0.1
 
        iter_num = 0
        while (iter_num < max_iter and self.get_dist(path_cur) > eps):            
            for i in range(len(path_cur)-2):
                for j in range(len(path_cur[0])):
                    grad = self.get_grad(path_cur,i+1,j) 
                    path_next[i+1][j] = path_cur[i+1][j] - L_rate*grad
                if (self.path_valid_check(path_next)):
                    path_cur = copy.deepcopy(path_next)
            iter_num += 1
        return path_cur
    def shortcut(self, path,num_iter):
        if len(path) < 2:
            return path
        for i in range (num_iter):
            i = random.randint(0,len(path)-2)
            j = random.randint(i+1,len(path)-1)
            if self.check_pair_valid(path[i],path[j]):
                if self.get_pathlenght([path[i],path[j]])<self.get_pathlenght(path[i:j+1]):
                    path = path[:i+1]+path[j:]
        return path
    def run(self, point_start, point_target, new_graph = True):
        path = []
        
        if new_graph:
            s = time()
            graph = Graph(self.n_dof)
            self.generate_samples(graph)
            print('PRM: Build the graph in {:.2f}s'.format(time() - s))
            with open('graph_map.pickle', 'wb') as f:
                    pickle.dump(graph, f, -1)
                    print('PRM: Graph is saved!')
        else:
            
            graph = Graph(self.n_dof)
            graph = pickle.load(open('graph_map.pickle', 'rb'))
            
        run = True
        start_joints = point_start
        graph.start_id = graph.add_node(start_joints)
        near_ids = graph.get_near_node_by_rad(start_joints,self.rad)
        print('PRM: Found neighbor {} with q_start'.format(len(near_ids)))
        if len(near_ids) == 0:
            run = False
        for near_node_id in near_ids:
            joints_n = graph.get_point(near_node_id)
            if self.check_pair_valid(start_joints,joints_n):
                graph.add_edge(graph.start_id,near_node_id)
        target_joints = point_target
        graph.target_id = graph.add_node(target_joints)
        near_ids = graph.get_near_node_by_rad(target_joints,self.rad)
        print('PRM: Found neighbor {} with q_start'.format(len(near_ids))) 
        if len(near_ids) == 0:
            run = False
        for near_node_id in near_ids:
            joints_n = graph.get_point(near_node_id)
            if self.check_pair_valid(target_joints,joints_n):
                graph.add_edge(graph.target_id,near_node_id)
        if run:
            path = self.search(graph)
            path_opt = self.grad_est(path)
            path_short = self.shortcut(path,50)
            path_opt_short = self.grad_est(path_short)
            return path,path_opt,path_short,path_opt_short
        return [],[],[],[]


