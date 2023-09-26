
from light_simulator import LightNetworkSimulator
from utils import *
from write2csv import *
import time
import multiprocessing
import logging


# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


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
#up_margin_list_vlc (C1), up_margin_list_radio (C2), negative_margin_count(disregard!), unused_ap_vlc, unused_ap_rf, #rate of VLC to radio (C6), #Distances (C7, C8) 

up_impact = [1, 1, 1, 1, 1, 0, 0]


down_impact = [1, 1, 1, 1, 1, 0, 0]

# Example usage
env_size = (100, 100, 3) #env_size_x, env_size_y, env_size_z
num_process = 64


def run_iteration(args):
    num_vlc_aps, num_rf_aps, num_nodes, count = args  # Unpack the arguments
    results = []
    results.extend([num_vlc_aps, num_rf_aps, num_nodes, count])
    start_time = time.time()
    # Create Network Simulator
    simulator = LightNetworkSimulator(num_vlc_aps, num_rf_aps, num_nodes, env_size)
    
    try:
        simulator.create_aps(APs_constants)
        simulator.create_nodes(env_size, nodes_constants)
        simulator.connect_nodes_to_aps()
        
        results_greedy = []
        results_greedy.extend([num_vlc_aps, num_rf_aps, num_nodes, count])
        C1, C2, C3, C4, C5, C6, C7, C8 = simulator.ggreedy()
        results_greedy.extend(C1)
        results_greedy.extend(C2)
        results_greedy.extend(C4)
        results_greedy.extend(C5)
        results_greedy.extend(C6)
        results_greedy.extend(C8)
        # Write results to a file
        write2file_sep2(results_greedy)
        
        up_bestcosts, up_best_combination, upcount, down_bestcosts, down_best_combination, downcount, unused_vlc_up, unused_rf_up, down_unused_vlc, down_unused_rf  = simulator.gen_all_topo_return_best_new( up_impact, down_impact)
        
        total_count = upcount * downcount if upcount is not None and downcount is not None else None
        
        # Record the end time
        end_time = time.time()
        
        # Calculate the execution time
        execution_time = end_time - start_time
        
        # Log the results
        logging.info(f"num_vlc_aps={num_vlc_aps}, num_rf_aps={num_rf_aps}, num_nodes={num_nodes}, count={count}")
        logging.info(f"down_best_combination: {down_best_combination}")
        logging.info(f"up_best_combination: {up_best_combination}")
        logging.info(f"Number of up topologies: {upcount}, down topologies: {downcount}, total topologies: {total_count}")
        logging.info(f"Execution time: {execution_time:.6f} seconds")
        
        results = addtoresults(up_bestcosts, down_bestcosts, results, 3* len(up_impact))
        results.extend([upcount, downcount, total_count, round(execution_time, 2), unused_vlc_up, unused_rf_up, down_unused_vlc, down_unused_rf])
        # Write results to a file
        write2file_sep(results)
        
    except Exception as e:
        # Handle exceptions and log errors
        logging.error(f"Error in simulation: {e}")


def main():
    vlc_aps_values = [2,4,6,8,10]
    rf_aps_values = [2,4,6,8,10]
    nodes_values = [5,10,15,20,25]
    run_count = 200

    # Create a multiprocessing pool before the loop
    pool = multiprocessing.Pool(processes = num_process)

    # Define a list of arguments for run_iteration
    args_list = [(num_vlc_aps, num_rf_aps, num_nodes, count)
                 for num_vlc_aps in vlc_aps_values
                 for num_rf_aps in rf_aps_values
                 for num_nodes in nodes_values
                 for count in range(run_count)]

    # Use pool.map to execute run_iteration in parallel
    pool.map(run_iteration, args_list)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main()
    
