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
        
    def ggreedy(self):
        uplinks = []
        downlinks = []
        for node in self.nodes:
            for ap in self.aps:
                if ap.node_name in node.uplink_options_all:
                    if (node.req_up_data_rate_gr != 0) and (ap.available_data_up >= node.req_up_data_rate_gr) :
                        ap.available_data_up -= node.req_up_data_rate_gr
                        uplinks.append((node.node_name, ap.node_name))
                        node.req_up_data_rate_gr = 0
                        node.uplink_options_gr.append(ap.node_name)
                if ap.node_name in node.downlink_options_all:
                    if (node.req_down_data_rate_gr != 0) and (ap.available_data_down >= node.req_down_data_rate_gr) :
                        ap.available_data_down -= node.req_down_data_rate_gr
                        downlinks.append(( ap.node_name, node.node_name))
                        node.req_down_data_rate_gr = 0   
                        node.downlink_options_gr.append(ap.node_name)
        down_margin_list_vlc = []
        down_margin_list_radio = []
        negative_margin_count = 0
        unused_ap_vlc = 0.
        unused_ap_rf = 0.
        
        # Calculate down margins for access points
        for ap in self.aps:
            if ap.available_data_down != ap.max_data_rate_tx:
                downmargin = ap.available_data_down
                if downmargin < 0:
                    negative_margin_count += 1
                if ap.nodetype == 'VLC':
                    down_margin_list_vlc.append(downmargin)
                else:
                    down_margin_list_radio.append(downmargin)
            else:
                if ap.nodetype == 'VLC':
                    unused_ap_vlc += 1
                else:
                    unused_ap_rf += 1

        C5 = self.calculate_statistics(down_margin_list_vlc)
        C6 = self.calculate_statistics(down_margin_list_radio)
        C7 = [negative_margin_count]
        C8 = [unused_ap_vlc, unused_ap_rf]
        up_margin_list_vlc = []
        up_margin_list_radio = []
        negative_margin_count = 0
        unused_ap_vlc = 0.
        unused_ap_rf = 0.
        
        # Calculate down margins for access points
        for ap in self.aps:
            if ap.available_data_up != ap.max_data_rate_rx:
                downmargin = ap.available_data_up
                if downmargin < 0:
                    negative_margin_count += 1
                if ap.nodetype == 'VLC':
                    up_margin_list_vlc.append(downmargin)
                else:
                    up_margin_list_radio.append(downmargin)
            else:
                if ap.nodetype == 'VLC':
                    unused_ap_vlc += 1
                else:
                    unused_ap_rf += 1
        C1 = self.calculate_statistics(up_margin_list_vlc)
        C2 = self.calculate_statistics(up_margin_list_radio)
        C3 = [negative_margin_count]
        C4 =  [unused_ap_vlc, unused_ap_rf]
        return C1, C2, C3, C4, C5, C6, C7, C8


    def calculate_statistics(self, data_list):
        if data_list:
            min_value = round(min(data_list),2)
            avg_value = round(sum(data_list) / len(data_list),2)
            max_value = round(max(data_list),2)
        else:
            min_value = avg_value = max_value = None  # Handle the case when the list is empty
        return [min_value, avg_value, max_value]


    def unused_ap_up(self, combin):
        used_vlc = {}
        used_rf = {}
        for link in combin:
            if link[1].startswith('v'):
                used_vlc[link[1]] = link[0]
            else:
                if link[1].startswith('r'):
                    used_rf[link[1]] = link[0]
                else:
                    print(f'Warning 1: Node: {link[0]}, AP: {link[1]}----------------')
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
                else:
                    print(f'Warning 1: Node: {link[1]}, AP: {link[0]}----------------')
        return self.num_vlc_aps - len(used_vlc), self.num_rf_aps - len(used_rf)
    
    
    
    def gen_all_topo_return_best_new(self, up_impact, down_impact):
        uplinks, downlinks, stat1, stat2 = self.gen_link_pissibilities_all()  #change to gen_link_pissibilities_all then all possiblities are produced
        if not stat1 or not stat2:
            return  None, None, None, None, None, None, None, None, None, None

        if stat1:
            up_topologies = LightTopology(self.nodes, self.aps, self.distances, self.up_data_rate, self.down_data_rate, up_impact)
            up_bestcosts, up_best_combination, upcount = up_topologies.up_connection(uplinks)
            unused_vlc_up, unused_rf_up = self.unused_ap_up(up_best_combination)
            if not stat2:
                return up_bestcosts, up_best_combination, upcount, None, None, None , unused_vlc_up, unused_rf_up, None, None

        if stat2:    
            down_topologies = LightTopology(self.nodes, self.aps, self.distances, self.up_data_rate, self.down_data_rate,  down_impact)
            down_bestcosts, down_best_combination, downcount = down_topologies.down_connection(downlinks)
            down_unused_vlc, down_unused_rf = self.unused_ap_down(down_best_combination)
            if not stat1:
                return None, None, None, down_bestcosts, down_best_combination, downcount, None, None, down_unused_vlc, down_unused_rf
            return up_bestcosts, up_best_combination, upcount, down_bestcosts, down_best_combination, downcount, unused_vlc_up, unused_rf_up, down_unused_vlc, down_unused_rf 
            
    def gen_link_pissibilities_all(self):
        uplink_possibilities = []
        downlink_possibilities = []
        stat1 = True
        stat2 = True
        for node in self.nodes:
            row1 = []
            row2 = []
            self.up_data_rate[node.node_name] = node.req_up_data_rate
            self.down_data_rate[node.node_name] = node.req_down_data_rate
            for ap in node.uplink_options_all:
                row1.append((node.node_name, ap))
            for ap in node.downlink_options_all:
                row2.append(( ap, node.node_name))
            
            if not row1:
                print(f'A node ({node.node_name} has no up connection!)')
                stat1 = False
            if not row2:
                print(f'A node ({node.node_name} has no down connection!)')
            
            uplink_possibilities.append(row1)
            downlink_possibilities.append(row2)
        return uplink_possibilities, downlink_possibilities, stat1, stat2
    
    def gen_link_pissibilities_gr(self):
        uplink_possibilities = []
        downlink_possibilities = []
        stat1 = True
        stat2 = True
        for node in self.nodes:
            row1 = []
            row2 = []
            self.up_data_rate[node.node_name] = node.req_up_data_rate
            self.down_data_rate[node.node_name] = node.req_down_data_rate
            for ap in node.uplink_options_gr:
                row1.append((node.node_name, ap))
            for ap in node.downlink_options_gr:
                row2.append(( ap, node.node_name))
            
            if not row1:
                print(f'A node ({node.node_name} has no up connection!)')
                stat1 = False
            if not row2:
                print(f'A node ({node.node_name} has no down connection!)')
                stat2 = False
            
            uplink_possibilities.append(row1)
            downlink_possibilities.append(row2)
        return uplink_possibilities, downlink_possibilities, stat1, stat2
    
    def create_aps(self, APs_constants):
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
        for i in range(self.num_nodes):
            node = Node(i, env_size, nodes_constants)
            self.nodes.append(node)
            
    def connect_nodes_to_aps(self):
        for node in self.nodes:
            for ap in self.aps:
                distance = node.calculate_distance(ap.location)
                
                if distance <= ap.tr_range:
                    node.downlink_options_all.append(ap.node_name)
                    self.distances[(ap.node_name, node.node_name)] = distance
                    
                if ap.nodetype == "VLC":
                    tr_range = node.node_tr_V
                else:
                    tr_range = node.node_tr_R

                if distance <= tr_range:
                    node.uplink_options_all.append(ap.node_name)
                    self.distances[(node.node_name, ap.node_name)] = distance
    
