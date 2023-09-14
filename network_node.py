
from utils import *

class Node:
    def __init__(self, node_id, env_size, nodes_constants):
        """
        Initialize a Node object with the given parameters.

            node_id (int): The unique identifier for the node.
            location (tuple): The (x, y, z) coordinates of the node's location.
            node_tr_V (float): Maximum transmission range for VLC communication.
            node_tr_R (float): Maximum transmission range for RF communication.
            req_down_data_rate (float): Required data rate for the node's downloading.
            req_up_data_rate (float): Required data rate for the node's uploading.
            mobility_factor (int): Defines the speed and chance of mobility of the node (Use Likert scale).
            rem_en (float): Remaining energy of the node.
            sec_req (bool): Indicates if security requirements are needed for communication.

        
        """
        self.node_id = node_id
        self.location = generate_random_location(env_size[0], env_size[1], env_size[2])
        self.node_tr_V =  random_number_in_range(nodes_constants.min_node_tr_V, nodes_constants.max_node_tr_V)
        self.node_tr_R = random_number_in_range(nodes_constants.min_node_tr_R, nodes_constants.max_node_tr_R)
        self.nodetype = "Nod"
        self.req_down_data_rate = random_number_in_range(nodes_constants.min_req_down_data_rate , nodes_constants.max_req_down_data_rate )
        self.req_up_data_rate = random_number_in_range(nodes_constants.min_req_up_data_rate, nodes_constants.max_req_up_data_rate)
        self.node_name = "n" + str(self.node_id)
        self.uplink_options = []
        self.downlink_options = []
        
    def calculate_distance(self, loc):
        x1, y1, z1 = self.location
        x2, y2, z2 = loc
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return round(distance, 2)