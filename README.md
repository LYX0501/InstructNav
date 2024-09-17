# InstructNav

Enabling robots to navigate following diverse language instructions in unexplored environments is an attractive goal for human-robot interaction. In this work, we propose InstructNav, a generic instruction navigation system. InstructNav makes the first endeavor to handle various instruction navigation tasks without any navigation training or pre-built maps. To reach this goal, we introduce **Dynamic Chain-of-Navigation (DCoN)** to unify the planning process for different types of navigation instructions. Furthermore, we propose **Multi-sourced Value Maps** to model key elements in instruction navigation so that linguistic DCoN planning can be converted into robot actionable trajectories. 

[InstructNav](https://github.com/LYX0501/InstructNav/blob/main/InstructNav.png)

With InstructNav, we complete the R2R-CE task in a zero-shot way for the first time and outperform many task-training methods. Besides, InstructNav also surpasses the previous SOTA method by 10.48% on the zero-shot Habitat ObjNav and by 86.34% on demand-driven navigation DDN. Real robot experiments on diverse indoor scenes further demonstrate our method's robustness in coping with the environment and instruction variations. Please refer to more details in our paper: 
![InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment](https://arxiv.org/abs/2406.04882).
## 🔥 News
- 2024.9.11: The HM3D objnav benchmark code is released.
- 2024.9.5: Our paper is accepted by CoRL 2024. Codes will be released in the recent.

### Dependency ###
Our project is based on the [habitat-sim](https://github.com/facebookresearch/habitat-sim?tab=readme-ov-file) and [habitat-lab](https://github.com/facebookresearch/habitat-lab). Please follow the guides to install them in your python environment. You can directly install the latest version of habitat-lab and habitat-sim. And make sure you have properly download the navigation scenes [(HM3D, MP3D)](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) and the episode dataset for both visual-language navigation (VLN-CE) and object navigation.

### Installation ###
Firstly, clone our repo as:
```
git clone https://github.com/LYX0501/InstructNav.git
cd InstructNav
pip install -r requirements.txt
```
Our method depends on an open-vocalbulary detection and segmentation model [GLEE](https://github.com/FoundationVision/GLEE). Please check the original repo or try to use the copy located in the ./thirdparty/ directory.
### 

### Prepare your GPT4 and GPT4V API Keys ###
Please prepare your keys for calling the API for large-language model and large vision-language model.
We prefer to use the GPT4 and GPT4V to do the inference. And our code follows the AzureOpenAI calling process.
Before running the benchmark, you should prepare for your own api-keys and api-endpoint and api-version. You can check the ./llm_utils/gpt_request.py for usage details.
```
export GPT4_API_BASE=<YOUR_GPT4_ENDPOINT>
export GPT4_API_KEY=<YOUR_GPT4_KEY>
export GPT4_API_DEPLOY=<GPT4_MODEL_NAME>
export GPT4_API_VERSION=<GPT4_MODEL_VERSION>
export GPT4V_API_BASE=<YOUR_GPT4V_ENDPOINT>
export GPT4V_API_KEY=<YOUR_GPT4V_KEY>
export GPT4V_API_DEPLOY=<GPT4V_MODEL_NAME>
export GPT4V_API_VERSION=<GPT4V_MODEL_VERSION>
```

### Running our Benchmark Code ###
If everything goes well, you can directly run the evaluation code for different navigation tasks.
For example, 
```
python objnav_benchmark.py
```
And all the episode results, intermediate results such as GPT4 input/output and value maps will be saved in /tmp/ directory. The real-time agent first-person-view image observation, depth and segmentation will be saved in the project root directory. Examples are shown below:
![test](https://github.com/user-attachments/assets/51a65b07-70e2-49f3-a850-815b0ec151d0)

https://github.com/user-attachments/assets/04e37b91-c524-4c51-86d1-8fb72325f612






## BibTex
Please cite our paper if you find it helpful :)
```
@misc{InstructNav,
      title={InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment}, 
      author={Yuxing Long and Wenzhe Cai and Hongcheng Wang and Guanqi Zhan and Hao Dong},
      year={2024},
      eprint={2406.04882},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
}
```
