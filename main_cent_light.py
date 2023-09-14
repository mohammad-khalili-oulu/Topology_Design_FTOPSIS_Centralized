
from light_simulator import LightNetworkSimulator
from utils import *
from write2csv import *
import time


class APs_constants:
    max_rx_data_rate_VLC = 500
    max_tx_data_rate_VLC = 1000
    max_rx_data_rate_RF = 100
    max_tx_data_rate_RF = 500
    
    min_RF_AP_tr = 90
    min_VLC_AP_tr = 10
    max_RF_AP_tr = 110
    max_VLC_AP_tr = 30
    

class nodes_constants:
    min_req_down_data_rate = 100
    max_req_down_data_rate = 120
    min_req_up_data_rate = 10 
    max_req_up_data_rate = 12

    min_node_tr_V = 10
    max_node_tr_V = 30

    min_node_tr_R = 30
    max_node_tr_R = 60





#for ecntrall Fuzzy TOPSIS approach

up_impact = [1, 1, 1, 1, 0, 0]


down_impact = [1, 1, 1, 1, 0, 0]

# Example usage
env_size = (100, 100, 3) #env_size_x, env_size_y, env_size_z



def main():
    # Define the list of numbers for VLC, RF, and  nodes to iterate over
    vlc_aps_values = [5]
    rf_aps_values = [2]
    nodes_values = [4,6]
    
    run_count = 2
    for num_vlc_aps in vlc_aps_values:
        for num_rf_aps in rf_aps_values:
            for num_nodes in nodes_values:
                for count in range(run_count):
                    start_time = time.time()
                    print(f"Running with num_vlc_aps={num_vlc_aps}, num_rf_aps={num_rf_aps}, num_nodes={num_nodes}, count={count}")
                    results = []
                    results.extend([num_vlc_aps, num_rf_aps, num_nodes, count])
                        
                    # Create  Network Simulator
                    simulator = LightNetworkSimulator(num_vlc_aps, num_rf_aps, num_nodes, env_size)
                    simulator.create_aps(APs_constants)
                    simulator.create_nodes(env_size, nodes_constants)
                    simulator.connect_nodes_to_aps()
                    
                    up_bestcosts, up_best_combination, upcount, down_bestcosts, down_best_combination, downcount, unused_vlc, unused_rf = simulator.gen_all_topo_return_best_new( up_impact, down_impact)
                    
                    print(f'down_best_combination: {down_best_combination}, up_best_combination: "{up_best_combination}')
                    total_count = upcount * downcount if upcount is not None and downcount is not None else None
                    
                    print(f"Number of up topologies: {upcount}, down topologies: {downcount} and total topologies: {total_count  }")
                    # Record the end time
                    end_time = time.time()

                    # Calculate the execution time
                    execution_time = end_time - start_time
                    print(f"Execution time: {execution_time:.6f} seconds")
                    
                    
                    results = addtoresults(up_bestcosts, down_bestcosts, results, 3* len(up_impact))
                    results.extend([upcount, downcount, total_count, round(execution_time, 2), unused_vlc, unused_rf])
                    # Write results to a file
                    write2file_sep(results)
                    
    
if __name__ == "__main__":
    main()
    
