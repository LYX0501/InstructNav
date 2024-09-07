# InstructNav

Enabling robots to navigate following diverse language instructions in unexplored environments is an attractive goal for human-robot interaction. In this work, we propose InstructNav, a generic instruction navigation system. InstructNav makes the first endeavor to handle various instruction navigation tasks without any navigation training or pre-built maps. To reach this goal, we introduce Dynamic Chain-of-Navigation (DCoN) to unify the planning process for different types of navigation instructions. Furthermore, we propose Multi-sourced Value Maps to model key elements in instruction navigation so that linguistic DCoN planning can be converted into robot actionable trajectories. 

![InstructNav](https://github.com/LYX0501/InstructNav/blob/main/InstructNav.png)

With InstructNav, we complete the R2R-CE task in a zero-shot way for the first time and outperform many task-training methods. Besides, InstructNav also surpasses the previous SOTA method by 10.48% on the zero-shot Habitat ObjNav and by 86.34% on demand-driven navigation DDN. Real robot experiments on diverse indoor scenes further demonstrate our method's robustness in coping with the environment and instruction variations.

## BibTex
Please cite our paper if you find it helpful :)
```
@misc{long2024instructnavzeroshotgenericinstruction,
      title={InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment}, 
      author={Yuxing Long and Wenzhe Cai and Hongcheng Wang and Guanqi Zhan and Hao Dong},
      year={2024},
      eprint={2406.04882},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
}
```
