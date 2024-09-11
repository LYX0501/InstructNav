import numpy as np
def preprocess_depth(depth:np.ndarray,lower_bound:float=0.1,upper_bound:float=4.9):
    depth[np.where((depth<lower_bound)|(depth>upper_bound))] = 0
    return depth
def preprocess_image(image:np.ndarray):
    return image