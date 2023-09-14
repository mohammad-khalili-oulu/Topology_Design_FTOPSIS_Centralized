from access_point import AccessPoint
from network_node import Node
from utils import *
from small_topology import LightTopology



class LightNetworkSimulator:
    def __init__(self, num_vlc_aps, num_rf_aps, num_nodes, env_size):
        
        self.aps = []
        self.env_size = env_size
        self.nodes = []
        self.num_vlc_aps = num_vlc_aps
        self.num_rf_aps = num_rf_aps
        self.num_nodes = num_nodes
        self.distances = {}
        self.up_data_rate = {}
        self.down_data_rate = {}
        
    def unused_ap_up(self, combin):
        used_vlc = {}
        used_rf = {}
        for link in combin:
            if link[1].startswith('v'):
                used_vlc[link[1]] = link[0]
            else:
                if link[1].startswith('r'):
                    used_rf[link[1]] = link[0]
                
                
        return self.num_vlc_aps - len(used_vlc), self.num_rf_aps - len(used_rf)
    
    def unused_ap_down(self, combin):
        used_vlc = {}
        used_rf = {}
        for link in combin:
            if link[0].startswith('v'):
                used_vlc[link[0]] = link[1]
            else:
                if link[0].startswith('r'):
                    used_rf[link[0]] = link[1]
                
        
        return self.num_vlc_aps - len(used_vlc), self.num_rf_aps - len(used_rf)
    
    def gen_all_topo_return_best_new(self, up_impact, down_impact):
        uplinks, downlinks, stat1, stat2 = self.gen_link_pissibilities()
        if not stat1 or not stat2:
            
            return  None, None, None, None, None, None, None, None
        
        if stat1:
            up_topologies = LightTopology(self.nodes, self.aps, self.distances, self.up_data_rate, self.down_data_rate, up_impact)
            up_bestcosts, up_best_combination, upcount = up_topologies.up_connection(uplinks)
            unused_vlc, unused_rf = self.unused_ap_up(up_best_combination)
            if not stat2:
                return up_bestcosts, up_best_combination, upcount, None, None, None , unused_vlc, unused_rf
        if stat2:    
            down_topologies = LightTopology(self.nodes, self.aps, self.distances, self.up_data_rate, self.down_data_rate,  down_impact)
            down_bestcosts, down_best_combination, downcount = down_topologies.down_connection(downlinks)
            down_unused_vlc, down_unused_rf = self.unused_ap_down(down_best_combination)
            if not stat1:
                return None, None, None, down_bestcosts, down_best_combination, downcount,  down_unused_vlc, down_unused_rf
            return up_bestcosts, up_best_combination, upcount, down_bestcosts, down_best_combination, downcount, min(unused_vlc, down_unused_vlc), min(unused_rf, down_unused_rf)
            
    def gen_link_pissibilities(self):
        uplink_possibilities = []
        downlink_possibilities = []
        stat1 = True
        stat2 = True
        for node in self.nodes:
            row1 = []
            row2 = []
            self.up_data_rate[node.node_name] = node.req_up_data_rate
            self.down_data_rate[node.node_name] = node.req_down_data_rate
            for ap in node.uplink_options:
                row1.append((node.node_name, ap))
            for ap in node.downlink_options:
                row2.append(( ap, node.node_name))
            if not row1:
                #warnings.warn("A node  has no uplink options ", UserWarning)
                stat1 = False
            if not row2:
                #warnings.warn("A node has no downlink options ", UserWarning)
                stat2 = False
            
            uplink_possibilities.append(row1)
            downlink_possibilities.append(row2)
        return uplink_possibilities, downlink_possibilities, stat1, stat2
            
    def create_aps(self, APs_constants):
        """
        Create the VLC and RF access points .
        """
        locations = self.distribute_APs(self.env_size[0], self.env_size[1], self.num_vlc_aps + self.num_rf_aps)
        for i, nodetype in enumerate(["VLC"] * self.num_vlc_aps + ["RF"] * self.num_rf_aps):
            ap = AccessPoint(i, nodetype, locations[i], APs_constants)
            self.aps.append(ap)

    def distribute_APs(self, x, y, N):
        rows = math.ceil(math.sqrt(N))
        cols = math.ceil(N / rows)
        node_spacing_x = x / cols
        node_spacing_y = y / rows

        nodes = []

        for i in range(rows):
            for j in range(cols):
                if len(nodes) < N:
                    # Calculate the position of the node
                    node_x = j * node_spacing_x
                    node_y = i * node_spacing_y
                    nodes.append((node_x, node_y, self.env_size[2]))

        return nodes
    
    def create_nodes(self, env_size, nodes_constants):
        """
        Create the  nodes .
        """
        for i in range(self.num_nodes):
            node = Node(i, env_size, nodes_constants)
            self.nodes.append(node)
            
    def connect_nodes_to_aps(self):
        
        for node in self.nodes:
            for ap in self.aps:
                distance = node.calculate_distance(ap.location)
                
                if distance <= ap.tr_range:
                    node.downlink_options.append(ap.node_name)
                    self.distances[(ap.node_name, node.node_name)] = distance
                    
                if ap.nodetype == "VLC":
                    tr_range = node.node_tr_V
                else:
                    tr_range = node.node_tr_R

                if distance <= tr_range:
                    node.uplink_options.append(ap.node_name)
                    self.distances[(node.node_name, ap.node_name)] = distance
    


    