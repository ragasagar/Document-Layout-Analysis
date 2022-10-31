from ast import Break
import detectron2
from detectron2.utils.logger import setup_logger

from utils.detectron_util import create_model
from utils.util import conf_matrix_calc
import pandas as pd
# from ImageEmbedding.feature_embedding import FeatueEmbedding

# from targeted_learning import create_model
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, shutil
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
import shutil
from docbank_loader import DocBankLoader, DocBankConverter
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser
from CBIR.src.DB import Database
from CBIR.src.evaluate import infer_subset
from CBIR.src.resnet import ResNetFeat
from utils.detectron_util import create_model, crop_images_classwise, crop_images_classwise_ground_truth, do_evaluate, remove_dataset
from utils.submodlib_util import Margin_Sampling, Random_wrapper
import pandas as pd
from utils.util import aug_train_subset, create_dir, create_new_query, find_missclassified_object, get_category_details, get_lakeset_images, get_original_images_path, remove_dir

import torch

torch.cuda.set_device(1)
parser = default_argument_parser()

parser.add_argument("--output_path",          default="flmi_test3", help="Output_path")
parser.add_argument("--strategy", default="cmi", help="subset selection strategy")
parser.add_argument("--total_budget",      default="500", type=int,  help="Total AL budget")
parser.add_argument("--budget",   default="10", type=int, help="selection budget")
parser.add_argument("--lake_size",   default="100", type=int, help="selection budget")
parser.add_argument("--train_size",   default="100", type=int, help="selection budget")
parser.add_argument("--category",   default="list", type=str, help="Targeted class")
parser.add_argument("--private_category",   default="text", type=str, help="Private Targeted class")
 
arg = parser.parse_args()
print(arg)
query_path = 'query_data_img/'+ arg.category;
private_query_path = 'query_data_img/'+arg.private_category
category = [arg.category];
private_category = [arg.private_category]

dataset_dir = ("../publaynet/publaynet/train5",
               "../publaynet/publaynet/train.json")
init_train_dataset_dir = ("../publaynet/intial_train_img/train_data_img",
                          "../publaynet/intial_train_img/train_targeted.json")
train_data_dirs = ("publaynet/train_data_img",
                   "publaynet/train_targeted.json")
# val_data_dirs = ("publaynet/train_data_img",
#                    "publaynet/train_targeted.json")
# test_data_dirs = ("publaynet/train_data_img",
#                    "publaynet/train_targeted.json")
lake_data_dirs = ("publaynet/lake_data_img",
                  "publaynet/lake_targeted.json")
test_data_dirs = ("../publaynet/test_data_img",
                  "../publaynet/test_targeted.json")
val_data_dirs = ("../publaynet/val_data_img",
                 "../publaynet/val_targeted.json")

train_path = 'model_result'
training_name = arg.output_path
model_path = os.path.join(train_path, training_name)
if (not os.path.exists(model_path)):
    create_dir(model_path)

# step 1
# train a faster_rcnn model on the initial_set
output_dir = os.path.join(model_path, "initial_training")
config_file_path = '../publaynet/configs/publaynet/faster_rcnn_R_101_FPN_3x.yaml'
prediction_score_threshold = 0.7
selection_strag = arg.strategy
selection_budget = arg.budget
budget = arg.total_budget
cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.DATASETS.TRAIN = ("initial_set",)
cfg.DATASETS.TEST = ('test_set', 'val_set')
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 6000
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.MODEL.RPN.NMS_THRESH = 0.8
cfg.TEST.EVAL_PERIOD = 1000
cfg.OUTPUT_DIR = output_dir
cfg.TRAINING_NAME = training_name

logger = setup_logger(os.path.join(output_dir, cfg.TRAINING_NAME))

# intial lake set images from intial dataset path
category_ratio = {
    "text": .71,
    "title": .19,
    "list": .024,
    "table": .031,
    "figure": 0.05
}

# getting the new lake set and train set
# get_lakeset_images(dataset_dir, lake_data_dirs, int(arg.lake_size), category_ratio)
# get_lakeset_images(init_train_dataset_dir,
#                    train_data_dirs, int(arg.train_size), category_ratio)

# given an initial_set and test_set dataset
register_coco_instances(
    "initial_set", {}, train_data_dirs[1], train_data_dirs[0])
register_coco_instances("test_set", {}, test_data_dirs[1], test_data_dirs[0])
register_coco_instances("val_set", {}, val_data_dirs[1], val_data_dirs[0])

logger.info("Starting Initial_set Training")
cfg.MODEL_WEIGHTS = '../publaynet/Initial_model_weight/model_final.pkl'
model = create_model(cfg)
# model.train()
logger.info("Initial_set training complete")

iteration = 100
result_val = []
result_test = []
# before starting the model active learning loop, calculating the embedding of the lake datset
cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
model = create_model(cfg, "test")

n_classes = 5
confusion_matrix = np.zeros([n_classes, n_classes])
metadata_train = MetadataCatalog.get("test_set")
data_dict = DatasetCatalog.get("test_set");
for d in  data_dict:
    img = cv2.imread(d["file_name"])
    outputs = model(img)
    labels = list()
    detections = list()
    for coord, conf, cls, ann in zip(
        outputs["instances"].get("pred_boxes").tensor.cpu().numpy(),
        outputs["instances"].get("scores").cpu().numpy(),
        outputs["instances"].get("pred_classes").cpu().numpy(),
        d["annotations"]
    ):
        labels.append([ann["category_id"]] + ann["bbox"])
        detections.append(list(coord) + [conf] + [cls])    
    confusion_matrix += conf_matrix_calc(np.array(labels), np.array(detections), n_classes, conf_thresh=0.5, iou_thresh=0.5)
matrix_indexes = ["text","title","list","table","figure"]
df = pd.DataFrame(confusion_matrix, columns=matrix_indexes, index=matrix_indexes)
df.to_csv("result.csv");

