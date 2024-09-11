import numpy as np
import quaternion
def habitat_camera_intrinsic(config):
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height, 'The configuration of the depth camera should be the same as rgb camera.'
    assert config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov, 'The configuration of the depth camera should be the same as rgb camera.'
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]],np.float32)
    return intrinsic_matrix

def habitat_translation(position):
    return np.array([position[0],position[2],position[1]])

def habitat_rotation(rotation):
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    transform_matrix = np.array([[1,0,0],
                                 [0,0,1],
                                 [0,1,0]])
    rotation_matrix = np.matmul(transform_matrix,rotation_matrix)
    return rotation_matrix
