
from network_node import *
from access_point import *
from utils import *
from ctypes import *


so_file = "./ftopsis.so"
c_functions = CDLL(so_file)
class CFuzzyNumber(Structure):
    _fields_ = [("a", c_double), ("b", c_double), ("c", c_double)]

class LightTopology:
    def __init__(self, nodes, aps, distances, up_data_rate, down_data_rate, impact):
        self.nodes = nodes
        self.aps = aps
        self.distances = distances
        self.up_data_rate = up_data_rate
        self.down_data_rate = down_data_rate
        self.num_criteria = len(impact)
        self.c_array_impact = (c_int * len(impact))(*impact)
        self.upcount = 0
        self.downcount = 0
        
        
    def generate_combinations(self, links, current_node, current_combination, best_combination, bestcosts, up_down):
        if current_node == len(links):
            if  bestcosts.get("best", -1) == -1:
                C1, C2, C3, C4, C5 , C6, C7 = self.all_costs(current_combination ,up_down)
                bestcosts["best"] = [C1, C2, C4, C5 , C6, C7]  # a topology considered and save it
                best_combination.clear()
                best_combination.extend(current_combination)
            else: 
                costs_of_two = []
                costs_of_two.append(bestcosts["best"])
                C1, C2, C3, C4, C5 , C6, C7 = self.all_costs(current_combination ,up_down)
                if C3 > 0:
                    return
                costs_of_two.append([C1, C2, C4, C5 , C6, C7] )
                costs = self.convert2c(costs_of_two, self.num_criteria, 2)
                best_cc = [0 for i in range(2)]
                c_array_cc = (c_double * len(best_cc))(*best_cc)
                c_functions.f_topsis(  2, self.num_criteria, costs, self.c_array_impact, c_array_cc)
                best_cc = list(c_array_cc)
                
                if best_cc[0] < best_cc[1]:
                    
                    bestcosts["best"] = costs_of_two[1]
                    best_combination.clear()
                    best_combination.extend(current_combination)
                    
            return

        for node, relation in links[current_node]:
            current_combination.append((node, relation))
            self.generate_combinations(links, current_node + 1, current_combination, best_combination, bestcosts, up_down)
            current_combination.pop()
            
            
    def up_connection(self, uplinks):
        best_combination = []
        bestcosts = { }
        self.generate_combinations(uplinks, 0, [], best_combination, bestcosts, 0)
        return bestcosts["best"], best_combination, self.upcount
    
    def down_connection(self, downlinks):
        best_combination = []
        bestcosts = { }
        self.generate_combinations(downlinks, 0, [], best_combination, bestcosts, 1)
        return bestcosts["best"], best_combination, self.downcount
        
    def convert2c(self, cost, num_criteria, num):
        cost2 = (CFuzzyNumber  * num_criteria * num)()
        for i in range(num):
            for j in range(num_criteria):
                
                cost2[i][j] = CFuzzyNumber(cost[i][j].a, cost[i][j].b, cost[i][j].c) 
        return cost2
        
    def all_costs(self, links, up_down):
        if up_down == 0:
            self.upcount += 1
            C1, C2, C3, C4 = self.up_margin_computation(links) #up_margin_list_vlc, up_margin_list_radio, negative_margin_count, unused_ap
            C5  = self.up_price_computation( links) #rate of VLC to radio
            C6, C7 = self.up_distance_computation(links) #Distances
            return C1, C2, C3, C4, C5 , C6, C7
        else:
            self.downcount += 1
            C1, C2, C3, C4 = self.down_margin_computation(links)
            C5  = self.down_price_computation( links)
            C6, C7 = self.down_distance_computation(links)
            return C1, C2, C3, C4, C5 , C6, C7
            
    def up_margin_computation(self, links):
        up_margin_list_vlc = []
        up_margin_list_radio = []
        negative_margin_count = 0
        used_data = {}
        unused_ap = 0
        # Calculate the used data for each node
        for upl in links:
            used_data[upl[1]] = used_data.get(upl[1], 0) - self.up_data_rate.get(upl[0], 0)

        # Calculate up margins for access points
        for ap in self.aps:
            if used_data.get(ap.node_name, 0) != 0:
                upmargin = ap.max_data_rate_rx + used_data.get(ap.node_name, 0)
                
                if upmargin < 0:
                    negative_margin_count += 1
                
                if ap.nodetype == 'VLC':
                    up_margin_list_vlc.append(upmargin)
                else:
                    up_margin_list_radio.append(upmargin)
            else:
                unused_ap += 1
        C1 = set2fuzzy(up_margin_list_vlc)
        C2 = set2fuzzy(up_margin_list_radio)
        #C3 = set2fuzzy([negative_margin_count])
        C3 = negative_margin_count
        C4 = set2fuzzy([unused_ap])
                    
        return C1, C2, C3, C4
    
    def down_margin_computation(self, links):
        down_margin_list_vlc = []
        down_margin_list_radio = []
        negative_margin_count = 0
        used_data = {}
        unused_ap = 0.
        
        # Calculate the used data for each node
        for dwl in links:
            used_data[dwl[0]] = used_data.get(dwl[0], 0) - self.down_data_rate.get(dwl[1], 0)

        # Calculate down margins for access points
        for ap in self.aps:
            if used_data.get(ap.node_name, 0) != 0:
                downmargin = ap.max_data_rate_tx + used_data.get(ap.node_name, 0)
                
                if downmargin < 0:
                    negative_margin_count += 1
                
                if ap.nodetype == 'VLC':
                    down_margin_list_vlc.append(downmargin)
                else:
                    down_margin_list_radio.append(downmargin)
            else:
                unused_ap += 1
        C1 = set2fuzzy(down_margin_list_vlc)
        C2 = set2fuzzy(down_margin_list_radio)
        #C3 = set2fuzzy([negative_margin_count])
        C3 = negative_margin_count
        C4 = set2fuzzy([unused_ap])
        return C1, C2, C3, C4
    
    
    def up_price_computation(self, uplinks):
        A5 = 0
        A6 = 0
        for upl in uplinks:
            if upl[1].startswith('v'):
                A5 += 1
            else:
                A6 += 1
        if A6 == 0:
            return fuzzy_number(9, 9, 9)
        
        ratio = A5 *100 / A6 
        if ratio > 80:
            return fuzzy_number(7, 9, 9)
        elif ratio >= 60:
            return fuzzy_number(5, 7, 9)
        elif ratio >= 40:
            return fuzzy_number(3, 5, 7)
        elif ratio >= 20:
            return fuzzy_number(1, 3, 5)
        else:
            return fuzzy_number(1, 1, 3)
        
            
            
    def down_price_computation(self, downlinks):
        A7 = 0
        A8 = 0
        for dwl in downlinks:
            if dwl[0].startswith('v'):
                A7 += 1
            else:
                A8 += 1
        if A8 == 0:
            return fuzzy_number(9, 9, 9)
        
        ratio = A7 *100 / A8 
        if ratio > 80:
            return fuzzy_number(7, 9, 9)
        elif ratio >= 60:
            return fuzzy_number(5, 7, 9)
        elif ratio >= 40:
            return fuzzy_number(3, 5, 7)
        elif ratio >= 20:
            return fuzzy_number(1, 3, 5)
        else:
            return fuzzy_number(1, 1, 3)
            
    def up_distance_computation(self, uplinks):
        A1 = []
        A3 = []
        for upl in uplinks:
            if upl[1].startswith('v'):
                A1.append(self.distances[(upl[0], upl[1])])
            else:
                A3.append(self.distances[(upl[0], upl[1])])
        C1 = set2fuzzy(A1)
        C3 = set2fuzzy(A3)
        return C1, C3

    def down_distance_computation(self,  downlinks):
        A2 = []
        A4 = []
        for dwl in downlinks:
            if dwl[0].startswith('v'):
                A2.append(self.distances[(dwl[0], dwl[1])])
            else:
                A4.append(self.distances[(dwl[0], dwl[1])])
        C2 = set2fuzzy(A2)
        C4 = set2fuzzy(A4)
        return  C2, C4