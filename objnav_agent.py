import habitat
import numpy as np
import cv2
import ast
import open3d as o3d
from mapping_utils.geometry import *
from mapping_utils.projection import *
from mapping_utils.path_planning import *
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from mapper import Instruct_Mapper
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from llm_utils.nav_prompt import CHAINON_PROMPT,GPT4V_PROMPT
from llm_utils.gpt_request import gpt_response,gptv_response

class HM3D_Objnav_Agent:
    def __init__(self,env:habitat.Env,mapper:Instruct_Mapper):
        self.env = env
        self.mapper = mapper
        self.episode_samples = 0
        self.planner = ShortestPathFollower(env.sim,0.5,False)

    def translate_objnav(self,object_goal):
        if object_goal.lower() == 'plant':
            return "Find the <%s>."%"potted_plant"
        elif object_goal.lower() == "tv_monitor":
            return "Find the <%s>."%"television_set"
        else:
            return "Find the <%s>."%object_goal
    
    def reset_debug_probes(self):
        self.rgb_trajectory = []
        self.depth_trajectory = []
        self.topdown_trajectory = []
        self.segmentation_trajectory = []

        self.gpt_trajectory = []
        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        
        self.obstacle_affordance_trajectory = []
        self.semantic_affordance_trajectory = []
        self.history_affordance_trajectory = []
        self.action_affordance_trajectory = []
        self.gpt4v_affordance_trajectory = []
        self.affordance_trajectory = []

    def reset(self):
        self.episode_samples += 1
        self.episode_steps = 0
        self.obs = self.env.reset()
        self.mapper.reset(self.env.sim.get_agent_state().sensor_states['rgb'].position,self.env.sim.get_agent_state().sensor_states['rgb'].rotation)
        self.instruct_goal = self.translate_objnav(self.env.current_episode.object_category)
        self.trajectory_summary = ""
        self.reset_debug_probes()     
       
    def rotate_panoramic(self,rotate_times = 12):
        self.temporary_pcd = []
        self.temporary_images = []
        for i in range(rotate_times):
            if self.env.episode_over:
                break
            self.update_trajectory()
            self.temporary_pcd.append(self.mapper.current_pcd)
            self.temporary_images.append(self.rgb_trajectory[-1])
            self.obs = self.env.step(3)
            
    def concat_panoramic(self,images):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10, 3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 != 0:
                row = (i//6)
                col = ((i%6)//2)
                copy_images[i] = cv2.putText(copy_images[i],"Direction %d"%i,(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
                background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,col*width + col * 10:col*width+col*10+width,:] = copy_images[i]
                
        return background_image
    
    def update_trajectory(self):
        self.episode_steps += 1
        self.metrics = self.env.get_metrics()
        self.rgb_trajectory.append(cv2.cvtColor(self.obs['rgb'],cv2.COLOR_BGR2RGB))
        self.depth_trajectory.append((self.obs['depth']/5.0 * 255.0).astype(np.uint8))
        
        topdown_image = cv2.cvtColor(colorize_draw_agent_and_fit_to_height(self.metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)
        topdown_image = cv2.putText(topdown_image,'Success:%.2f,SPL:%.2f,SoftSPL:%.2f,DTS:%.2f'%(self.metrics['success'],self.metrics['spl'],self.metrics['soft_spl'],self.metrics['distance_to_goal']),(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)
        self.topdown_trajectory.append(topdown_image)
        
        self.position = self.env.sim.get_agent_state().sensor_states['rgb'].position
        self.rotation = self.env.sim.get_agent_state().sensor_states['rgb'].rotation

        self.mapper.update(self.rgb_trajectory[-1],self.obs['depth'],self.position,self.rotation)
        self.segmentation_trajectory.append(self.mapper.segmentation)
        self.observed_objects = self.mapper.get_appeared_objects()

        cv2.imwrite("monitor-rgb.jpg",self.rgb_trajectory[-1])
        cv2.imwrite("monitor-depth.jpg",self.depth_trajectory[-1])
        cv2.imwrite("monitor-segmentation.jpg",self.segmentation_trajectory[-1])
            
    def save_trajectory(self,dir="./tmp_objnav/"):
        import imageio
        import os
        os.makedirs(dir)

        self.mapper.save_pointcloud_debug(dir) 
        fps_writer = imageio.get_writer(dir+"fps.mp4", fps=4)
        dps_writer = imageio.get_writer(dir+"depth.mp4", fps=4)
        seg_writer = imageio.get_writer(dir+"segmentation.mp4", fps=4)
        metric_writer = imageio.get_writer(dir+"metrics.mp4",fps=4)
        for i,img,dep,seg,met in zip(np.arange(len(self.rgb_trajectory)),self.rgb_trajectory,self.depth_trajectory,self.segmentation_trajectory,self.topdown_trajectory):
            fps_writer.append_data(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            dps_writer.append_data(dep)
            seg_writer.append_data(cv2.cvtColor(seg,cv2.COLOR_BGR2RGB))
            metric_writer.append_data(cv2.cvtColor(met,cv2.COLOR_BGR2RGB))

        for index,pano_img in enumerate(self.panoramic_trajectory):
            cv2.imwrite(dir+"%d-pano.jpg"%index,pano_img)
        with open(dir+"gpt4_history.txt",'w') as file:
            file.write("".join(self.gpt_trajectory))
        with open(dir+"gpt4v_history.txt",'w') as file:
            file.write("".join(self.gptv_trajectory))

        for i,afford,safford,hafford,cafford,gafford,oafford in zip(np.arange(len(self.affordance_trajectory)),self.affordance_trajectory,self.semantic_affordance_trajectory,self.history_affordance_trajectory,self.action_affordance_trajectory,self.gpt4v_affordance_trajectory,self.obstacle_affordance_trajectory):
            o3d.io.write_point_cloud(dir+"afford-%d-plan.ply"%i,afford)
            o3d.io.write_point_cloud(dir+"semantic-afford-%d-plan.ply"%i,safford)
            o3d.io.write_point_cloud(dir+"history-afford-%d-plan.ply"%i,hafford)
            o3d.io.write_point_cloud(dir+"action-afford-%d-plan.ply"%i,cafford)
            o3d.io.write_point_cloud(dir+"gpt4v-afford-%d-plan.ply"%i,gafford)
            o3d.io.write_point_cloud(dir+"obstacle-afford-%d-plan.ply"%i,oafford)

        fps_writer.close()
        dps_writer.close()
        seg_writer.close()
        metric_writer.close()
    
    def query_chainon(self):
        semantic_clue = {'observed object':self.observed_objects}
        query_content = "<Navigation Instruction>:{}, <Previous Plan>:{}, <Semantic Clue>:{}".format(self.instruct_goal,"{" + self.trajectory_summary + "}",semantic_clue)
        self.gpt_trajectory.append("Input:\n%s \n"%query_content)
        for i in range(10):
            try:
                raw_answer = gpt_response(query_content,CHAINON_PROMPT)
                print("GPT-4 Output Response: %s"%raw_answer)
                answer = raw_answer.replace(" ","")
                answer = answer[answer.index("{"):answer.index("}")+1]
                answer = ast.literal_eval(answer)
                if 'Action' in answer.keys() and 'Landmark' in answer.keys() and 'Flag' in answer.keys():
                    break
            except:
                continue
        self.gpt_trajectory.append("\nGPT-4 Answer:\n%s"%raw_answer)
        if self.trajectory_summary == "":
            self.trajectory_summary = self.trajectory_summary + str(answer['Action']) + '-' + str(answer['Landmark'])
        else:
            self.trajectory_summary = self.trajectory_summary + '-' + str(answer['Action']) + '-' + str(answer['Landmark'])
        return answer
    
    def query_gpt4v(self):
        images = self.temporary_images
        inference_image = self.concat_panoramic(images)
        cv2.imwrite("monitor-panoramic.jpg",inference_image)
        text_content = "<Navigation Instruction>:{}\n <Sub Instruction>:{}".format(self.instruct_goal,self.trajectory_summary.split("-")[-2] + "-" + self.trajectory_summary.split("-")[-1])
        self.gptv_trajectory.append("\nInput:\n%s \n"%text_content)
        for i in range(10):
            try:
                raw_answer = gptv_response(text_content,inference_image,GPT4V_PROMPT)
                print("GPT-4V Output Response: %s"%raw_answer)
                answer = raw_answer[raw_answer.index("Judgement: Direction"):]
                answer = answer.replace(" ","")
                answer = int(answer.split("Direction")[-1])
                break
            except:
                continue
        self.gptv_trajectory.append("GPT-4V Answer:\n%s"%raw_answer)
        self.panoramic_trajectory.append(inference_image)
        try:
            return answer
        except:
            return np.random.randint(0,12)
    
    def make_plan(self,rotate=True,failed=False):
        if rotate == True:
            self.rotate_panoramic()
        self.chainon_answer = self.query_chainon()
        self.gpt4v_answer = self.query_gpt4v()
        self.gpt4v_pcd = o3d.t.geometry.PointCloud(self.mapper.pcd_device)
        self.gpt4v_pcd = gpu_merge_pointcloud(self.gpt4v_pcd,self.temporary_pcd[self.gpt4v_answer])
        self.found_goal = bool(self.chainon_answer['Flag'])
        self.affordance_pcd,self.colored_affordance_pcd = self.mapper.get_objnav_affordance_map(self.chainon_answer['Action'],self.chainon_answer['Landmark'],self.gpt4v_pcd,self.chainon_answer['Flag'],failure_mode=failed)
        self.semantic_afford,self.history_afford,self.action_afford,self.gpt4v_afford,self.obs_afford = self.mapper.get_debug_affordance_map(self.chainon_answer['Action'],self.chainon_answer['Landmark'],self.gpt4v_pcd)
        
        self.affordance_map,self.colored_affordance_map = project_costmap(self.mapper.navigable_pcd,self.affordance_pcd,self.mapper.grid_resolution)
        self.target_point = self.mapper.navigable_pcd.point.positions[self.affordance_pcd.argmax()].cpu().numpy()
        self.plan_position = self.mapper.current_position.copy()
        target_index = translate_point_to_grid(self.mapper.navigable_pcd,self.target_point,self.mapper.grid_resolution)
        start_index = translate_point_to_grid(self.mapper.navigable_pcd,self.mapper.current_position,self.mapper.grid_resolution)
        self.path = path_planning(self.affordance_map,start_index,target_index)
        self.path = [translate_grid_to_point(self.mapper.navigable_pcd,np.array([[waypoint.y,waypoint.x,0]]),self.mapper.grid_resolution)[0] for waypoint in self.path]
        if len(self.path) == 0:
            self.waypoint = self.mapper.navigable_pcd.point.positions.cpu().numpy()[np.argmax(self.affordance_pcd)]
            self.waypoint[2] = self.mapper.current_position[2]
        elif len(self.path) < 5: 
            self.waypoint = self.path[-1]
            self.waypoint[2] = self.mapper.current_position[2]
        else:
            self.waypoint = self.path[4]
            self.waypoint[2] = self.mapper.current_position[2]

        self.affordance_trajectory.append(self.colored_affordance_pcd)
        self.obstacle_affordance_trajectory.append(self.obs_afford)
        self.semantic_affordance_trajectory.append(self.semantic_afford)
        self.history_affordance_trajectory.append(self.history_afford)
        self.action_affordance_trajectory.append(self.action_afford)
        self.gpt4v_affordance_trajectory.append(self.gpt4v_afford)
    
    def step(self):
        to_target_distance = np.sqrt(np.sum(np.square(self.mapper.current_position - self.waypoint)))
        if to_target_distance < 0.6 and len(self.path) > 0:
            self.path = self.path[min(5,len(self.path)-1):]
            if len(self.path) < 3:
                self.waypoint = self.path[-1]
                self.waypoint[2] = self.mapper.current_position[2]
            else:
                self.waypoint = self.path[2]
                self.waypoint[2] = self.mapper.current_position[2]

        pid_waypoint = self.waypoint + self.mapper.initial_position
        pid_waypoint = np.array([pid_waypoint[0],self.env.sim.get_agent_state().position[1],pid_waypoint[1]])
        act = self.planner.get_next_action(pid_waypoint)
        move_distance =  np.sqrt(np.sum(np.square(self.mapper.current_position - self.plan_position)))
        if (act == 0 or move_distance > 3.0) and not self.found_goal:
            self.make_plan(rotate=True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            pid_waypoint = np.array([pid_waypoint[0],self.env.sim.get_agent_state().position[1],pid_waypoint[1]])
            act = self.planner.get_next_action(pid_waypoint)
        if act == 0 and not self.found_goal:
            self.make_plan(False,True)
            pid_waypoint = self.waypoint + self.mapper.initial_position
            pid_waypoint = np.array([pid_waypoint[0],self.env.sim.get_agent_state().position[1],pid_waypoint[1]])
            act = self.planner.get_next_action(pid_waypoint)
            print("Warning: Failure locomotion and action = %d"%act)
        if not self.env.episode_over:
            self.obs = self.env.step(act)
            self.update_trajectory()
       
