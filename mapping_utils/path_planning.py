import numpy as np
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from .projection import *

def path_planning(costmap,start_index,goal_index):
    planmap = costmap.copy()
    planmap[planmap == 1] = 10
    grid = Grid(matrix=(planmap*100).astype(np.int32))
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    start_index[0][1] = np.clip(start_index[0][1],0,costmap.shape[1]-1)
    start_index[0][0] = np.clip(start_index[0][0],0,costmap.shape[0]-1)
    goal_index[0][1] = np.clip(goal_index[0][1],0,costmap.shape[1]-1)
    goal_index[0][0] = np.clip(goal_index[0][0],0,costmap.shape[0]-1)
    start = grid.node(start_index[0][1],start_index[0][0])
    goal = grid.node(goal_index[0][1],goal_index[0][0])
    path,_ = finder.find_path(start,goal,grid)
    return path

def visualize_path(costmap,path):
    visualize_costmap = costmap.copy()
    for waypoint in path:
        x = waypoint.y
        y = waypoint.x
        visualize_costmap[x,y] = 10
    visualize_costmap = cv2.resize(visualize_costmap,(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST)
    visualize_costmap = cv2.applyColorMap((255*visualize_costmap/10).astype(np.uint8),cv2.COLORMAP_JET)
    return visualize_costmap    

    
    
    
    
    
    
    
    