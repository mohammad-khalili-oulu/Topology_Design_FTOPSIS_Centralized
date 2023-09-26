from utils import  random_number_in_range

class AccessPoint:
    def __init__(self, ap_id, nodetype, location,  APs_constants):
        """
        Initialize an Access Point object.

        Parameters:
            ap_id (str): Access Point identifier.
            nodetype (str): Type of the Access Point (e.g., VLC, RF).
        """
        self.node_id = ap_id
        self.location = location
        self.max_tr = APs_constants.max_RF_AP_tr if nodetype == "RF" else APs_constants.max_VLC_AP_tr
        self.tr_range = random_number_in_range(APs_constants.min_RF_AP_tr if nodetype == "RF" else APs_constants.min_VLC_AP_tr, self.max_tr)
        self.max_data_rate_rx = APs_constants.max_rx_data_rate_RF if nodetype == "RF" else APs_constants.max_rx_data_rate_VLC
        self.max_data_rate_tx = APs_constants.max_tx_data_rate_RF if nodetype == "RF" else APs_constants.max_tx_data_rate_VLC
        self.nodetype = nodetype
        self.node_name = self.typemap() + str(self.node_id)
        self.available_data_down = self.max_data_rate_tx
        self.available_data_up = self.max_data_rate_rx
        

    def typemap(self):
        """
        Map the Access Point node type to a single character code.

        Returns:
            str: The single character code representing the node type ('v' for VLC, 'r' for RF, or 'unknown' if not recognized).
        """
        if self.nodetype == "VLC":
            return "v"
        elif self.nodetype == "RF":
            return "r"
        else:
            return "unknown"
