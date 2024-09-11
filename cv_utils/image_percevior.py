from constants import *
from .glee_detector import *
class GLEE_Percevior:
    def __init__(self,
                 glee_config=GLEE_CONFIG_PATH,
                 glee_checkpoint=GLEE_CHECKPOINT_PATH,
                 device = "cuda:0"):
        self.device = device
        self.glee_model = initialize_glee(glee_config,glee_checkpoint,device)
    def perceive(self,image,confidence_threshold=0.25,area_threshold=2500):
        pred_bboxes, pred_masks, pred_class, pred_confidence = glee_segmentation(image,self.glee_model,threshold_select=confidence_threshold,device=self.device)
        try:
            mask_area = np.array([mask.sum() for mask in pred_masks])
            bbox_trust = np.array([(bbox[0] > 20) & (bbox[2] < image.shape[1] - 20) for bbox in pred_bboxes])
            visualization = visualize_segmentation(image,pred_class[(mask_area>area_threshold) & bbox_trust],pred_masks[(mask_area>area_threshold) & bbox_trust])
            return pred_class[(mask_area>area_threshold) & bbox_trust],pred_masks[(mask_area>area_threshold) & bbox_trust],pred_confidence[(mask_area>area_threshold) & bbox_trust],[visualization]
        except:
            return [],[],[],[image]
