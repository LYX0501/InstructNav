import numpy as np
import open3d as o3d
import cv2
# obstacle = 0
# unknown = 1
# position = 2
# navigable = 3
# frontier = 4

def project_frontier(obstacle_pcd,navigable_pcd,obstacle_height=-0.7,grid_resolution=0.25):
    np_obstacle_points = obstacle_pcd.point.positions.cpu().numpy()
    np_navigable_points = navigable_pcd.point.positions.cpu().numpy()
    np_all_points = np.concatenate((np_obstacle_points,np_navigable_points),axis=0)
    max_bound = np.max(np_all_points,axis=0)
    min_bound = np.min(np_all_points,axis=0)
    grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
    grid_map = np.ones((grid_dimensions[0],grid_dimensions[1]),dtype=np.int32)
    # get navigable occupancy
    navigable_points = np_navigable_points
    navigable_indices = np.floor((navigable_points - min_bound) / grid_resolution).astype(int)
    navigable_indices[:,0] = np.clip(navigable_indices[:,0],0,grid_dimensions[0]-1)
    navigable_indices[:,1] = np.clip(navigable_indices[:,1],0,grid_dimensions[1]-1)
    navigable_indices[:,2] = np.clip(navigable_indices[:,2],0,grid_dimensions[2]-1)
    navigable_voxels = np.zeros(grid_dimensions,dtype=np.int32)
    navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = 1
    navigable_map = (navigable_voxels.sum(axis=2) > 0)
    grid_map[np.where(navigable_map>0)] = 3
    # get obstacle occupancy
    obstacle_points = np_obstacle_points
    obstacle_indices = np.floor((obstacle_points - min_bound) / grid_resolution).astype(int)
    obstacle_indices[:,0] = np.clip(obstacle_indices[:,0],0,grid_dimensions[0]-1)
    obstacle_indices[:,1] = np.clip(obstacle_indices[:,1],0,grid_dimensions[1]-1)
    obstacle_indices[:,2] = np.clip(obstacle_indices[:,2],0,grid_dimensions[2]-1)
    obstacle_voxels = np.zeros(grid_dimensions,dtype=np.int32)
    obstacle_voxels[obstacle_indices[:,0],obstacle_indices[:,1],obstacle_indices[:,2]] = 1
    obstacle_map = (obstacle_voxels.sum(axis=2) > 0)
    grid_map[np.where(obstacle_map>0)] = 0
     # get outer-border of navigable areas
    outer_border_navigable = ((grid_map == 3)*255).astype(np.uint8)
    contours,hierarchiy = cv2.findContours(outer_border_navigable,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    outer_border_navigable = cv2.drawContours(np.zeros((grid_map.shape[0],grid_map.shape[1])),contours,-1,(255,255,255),1).astype(np.float32)
    obstacles = ((grid_map == 0)*255).astype(np.float32)
    obstacles = cv2.dilate(obstacles.astype(np.uint8),np.ones((3,3)))
    outer_border_navigable = ((outer_border_navigable - obstacles) > 0)
    grid_map_x,grid_map_y = np.where(outer_border_navigable>0)
    grid_indexes = np.stack((grid_map_x,grid_map_y,obstacle_height*np.ones((grid_map_x.shape[0],))),axis=1)
    frontier_points = grid_indexes * grid_resolution + min_bound
    return frontier_points
    
def translate_grid_to_point(pointcloud,grid_indexes,grid_resolution=0.25):
    np_all_points = pointcloud.point.positions.cpu().numpy()
    min_bound = np.min(np_all_points,axis=0)
    translate_points = grid_indexes * grid_resolution + min_bound
    return translate_points

def translate_point_to_grid(pointcloud,point_poses,grid_resolution=0.25):
    if len(point_poses.shape) == 1:
        point_poses = point_poses[np.newaxis,:]
    np_all_points = pointcloud.point.positions.cpu().numpy()
    min_bound = np.min(np_all_points,axis=0)
    grid_index = np.floor((point_poses - min_bound) / grid_resolution).astype(int)
    return grid_index[:,0:2]

def project_costmap(navigable_pcd,affordance_value,grid_resolution=0.25):
    navigable_points = navigable_pcd.point.positions.cpu().numpy()
    max_bound = np.max(navigable_points,axis=0)
    min_bound = np.min(navigable_points,axis=0)
    grid_dimensions = np.ceil((max_bound - min_bound) / grid_resolution).astype(int)
    navigable_voxels = np.zeros(grid_dimensions,dtype=np.float32)
    navigable_indices = np.floor((navigable_points - min_bound) / grid_resolution).astype(int)
    navigable_indices[:,0] = np.clip(navigable_indices[:,0],0,grid_dimensions[0]-1)
    navigable_indices[:,1] = np.clip(navigable_indices[:,1],0,grid_dimensions[1]-1)
    navigable_indices[:,2] = np.clip(navigable_indices[:,2],0,grid_dimensions[2]-1)
    navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = affordance_value
    navigable_costmap = navigable_voxels.max(axis=2)
    navigable_costmap = 1 - navigable_costmap
    color_navigable_costmap = cv2.applyColorMap((navigable_costmap*255).astype(np.uint8),cv2.COLORMAP_JET)
    color_navigable_costmap = cv2.resize(color_navigable_costmap,(0,0),fx=5,fy=5,interpolation=cv2.INTER_NEAREST)
    return navigable_costmap,color_navigable_costmap

