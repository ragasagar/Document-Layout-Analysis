from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    build_detection_test_loader,
   
)
from detectron2.modeling import build_model
from collections import OrderedDict
from detectron2.evaluation import (
        COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds, setup_logger
import logging
from detectron2.engine import DefaultPredictor


logger = logging.getLogger("embedding")

# img = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# #Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# emb_model = resnet50(weights=weights)
# emb_model.eval()
# 
class GetObjectEmbeddings:
    def __init__():
        weights = ResNet50_Weights.DEFAULT
        self.emb_model=emb_model = resnet50(weights=weights)
        self.emb_model.eval()
        output_dir="model_result/"
        trained_model_weights_path = "../publaynet/['model_result/test/model_final.pth"
        config_file_path = '../publaynet/configs/publaynet/faster_rcnn_R_101_FPN_3x.yaml'
        prediction_score_threshold = 0.7
        cfg = get_cfg()
        cfg.merge_from_file(config_file_path)
        cfg.DATASETS.TEST = ('seg_test',)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold   # set a custom testing threshold

        cfg.MODEL.WEIGHTS = trained_model_weights_path
        self.obj_model = build_model(cfg)
    
    def get_objs(image):
        

        

