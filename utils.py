import random
import math


class fuzzy_number:
    def __init__(self, num1=0, num2=0, num3=0):
        self.a = num1
        self.b = num2
        self.c = num3
        
    
def fuzzy2str(num):
    str1 = '(' + str(round(num.a, 2)) + ',' + str(round(num.b, 2)) + ',' + str(round(num.c, 2)) + ')'
    return str1
        
        
def set2fuzzy( listl):
    if listl:
        minl = min(listl)
        avg = sum(listl) / len(listl)
        maxl = max(listl)
    else:
        minl = 0
        avg = 0
        maxl = 0
    f = fuzzy_number(minl, avg, maxl)
    return f

def generate_random_location(env_size_x, env_size_y, env_size_z):
    """
    Generate random (x, y, z) coordinates within the square boundaries.

    Args:
        env_size (float): The size of the environment.

    Returns:
        tuple: A tuple representing the random (x, y, z) coordinates.
    """
    x = random.uniform(0, env_size_x)
    y = random.uniform(0, env_size_y)
    z = random.uniform(0, env_size_z)
    return x, y, z





def random_number_in_range(lower_bound, upper_bound):
    """
    Generate a random number within the specified range.

    Args:
        lower_bound (float): The lower bound of the range.
        upper_bound (float): The upper bound of the range.

    Returns:
        float: A random number within the specified range.
    """
    return random.uniform(lower_bound, upper_bound)

