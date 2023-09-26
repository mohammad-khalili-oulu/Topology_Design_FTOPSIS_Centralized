import csv
from utils import *
import multiprocessing

# Create a Lock object for file access
file_lock = multiprocessing.Lock()
        
def addtoresults(up_bestcosts, down_bestcosts,results,x):
    up_criteria = []
    if up_bestcosts is not None:
        for e in up_bestcosts:
            up_criteria.append(round(e.a,2))
            up_criteria.append(round(e.b,2))
            up_criteria.append(round(e.c,2))
    else:
        up_criteria.extend([ None for i in range(x)] )    
    down_criteria = []
    if down_bestcosts is not None:
        for e in down_bestcosts:
            down_criteria.append(round(e.a,2))
            down_criteria.append(round(e.b,2))
            down_criteria.append(round(e.c,2))
    else:
        down_criteria.extend([ None for i in range(x)] )  
    
    results.extend(up_criteria)
    results.extend(down_criteria)
    return results

def write2file_sep( data):
    
    
    # Specify the file path
    csv_file_path = 'output.csv'

    # Write data to the CSV file
    with file_lock:
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            writer.writerows([data])

def write2file_sep2( data):
    
    
    # Specify the file path
    csv_file_path = 'outputgreedy.csv'

    # Write data to the CSV file
    with file_lock:
        with open(csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            writer.writerows([data])