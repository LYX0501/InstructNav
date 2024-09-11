from cv_utils.object_list import categories
GLEE_CONFIG_PATH = "./thirdparty/GLEE/configs/SwinL.yaml"
GLEE_CHECKPOINT_PATH = "./thirdparty/GLEE/GLEE_SwinL_Scaleup10m.pth"
DETECT_OBJECTS = [[cat['name'].lower()] for cat in categories]
INTEREST_OBJECTS = ['bed','chair','toilet','potted_plant','television_set','sofa']


