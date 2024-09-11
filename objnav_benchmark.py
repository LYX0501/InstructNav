import habitat
import os
import argparse
import csv
from tqdm import tqdm
from config_utils import hm3d_config,mp3d_config
from mapping_utils.transform import habitat_camera_intrinsic
from mapper import Instruct_Mapper
from objnav_agent import HM3D_Objnav_Agent
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
def write_metrics(metrics,path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=500)
    parser.add_argument("--mapper_resolution",type=float,default=0.05)
    parser.add_argument("--path_resolution",type=float,default=0.2)
    parser.add_argument("--path_scale",type=int,default=5)
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    args = get_args()
    habitat_config = hm3d_config(stage='val',episodes=args.eval_episodes)
    habitat_env = habitat.Env(habitat_config)
    habitat_mapper = Instruct_Mapper(habitat_camera_intrinsic(habitat_config),
                                    pcd_resolution=args.mapper_resolution,
                                    grid_resolution=args.path_resolution,
                                    grid_size=args.path_scale)
    habitat_agent = HM3D_Objnav_Agent(habitat_env,habitat_mapper)
    evaluation_metrics = []
    for i in tqdm(range(args.eval_episodes)):
        habitat_agent.reset()
        habitat_agent.make_plan()
        while not habitat_env.episode_over and habitat_agent.episode_steps < 495:
            habitat_agent.step()
        habitat_agent.save_trajectory("./tmp/episode-%d/"%i)
        evaluation_metrics.append({'success':habitat_agent.metrics['success'],
                                'spl':habitat_agent.metrics['spl'],
                                'distance_to_goal':habitat_agent.metrics['distance_to_goal'],
                                'object_goal':habitat_agent.instruct_goal})
        write_metrics(evaluation_metrics)