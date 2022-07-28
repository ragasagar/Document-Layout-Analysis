from ast import Break
import detectron2
from detectron2.utils.logger import setup_logger
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


from utils.utils import *
from utils.coco_util import *


#initial weight and config path
config_file_path = 'configs/coco/faster_rcnn_R_101_FPN_3x.yaml'
# weights_paths = ["new_model_weights/model_final.pth",
#     'random_model_weights/model_final.pth',
#     'com_model_weights/model_final.pth',
#     'fl2mi_model_weights/model_final.pth',
#     'flcmi_new_weight/model_final.pth'
# ]


weights_paths = ['final_model_testing/model_final.pth'];
## intial initializeation of the parameters
register_coco_instances("docbank_seg_train",{}, "COCOTrainData.json", ".")
# print(MetadataCatalog.get("train_data_dcoco"))
register_coco_instances("docbank_seg_val",{}, "PASCAL_VOC/PASCAL_VOC/val_targeted.json", ".")
final_data = []
for weights_path in weights_paths:
    cfg = get_cfg();
    cfg.DATASETS.TRAIN = ("docbank_seg_train",)
    cfg.DATASETS.TEST = ("docbank_seg_test",)

    cfg.merge_from_file(config_file_path)
    # cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    predictor = DefaultPredictor(cfg)
    image = cv2.imread('val_data_img/000024.jpg')

    print(predictor(image))

#     evaluator = COCOEvaluator("docbank_seg_val", output_dir="final_new_weights")
#     val_loader = build_detection_test_loader(cfg, "docbank_seg_val")
#     result = inference_on_dataset(predictor.model, val_loader, evaluator)
#     print(find_missclassified_object(result))
#     for k , val  in  result.items():
#         temp = list(val.keys())
#         final_data.append(list(val.values()))
#         break;
# print(temp)
# csv = pd.DataFrame(final_data, columns=temp, index=weights_paths)

# csv.to_csv("test.csv")






# n_classes = 13
# confusion_matrix = np.zeros([n_classes + 1, n_classes + 1])
# dataset_dicts_validation = DatasetCatalog.get("docbank_seg_test")
# metadata_train = MetadataCatalog.get("docbank_seg_train")
# for d in dataset_dicts_validation:
#     img = cv2.imread(d["file_name"])
#     outputs = predictor(img)
#     labels = list()
#     detections = list()
#     for coord, conf, cls, ann in zip(
#         outputs["instances"].get("pred_boxes").tensor.cpu().numpy(),
#         outputs["instances"].get("scores").cpu().numpy(),
#         outputs["instances"].get("pred_classes").cpu().numpy(),
#         d["annotations"]
#     ):
#         labels.append([ann["category_id"]] + ann["bbox"])
#         detections.append(list(coord) + [conf] + [cls])    
#     confusion_matrix += conf_matrix_calc(np.array(labels), np.array(detections), n_classes, conf_thresh=0.4, iou_thresh=0.6)
# matrix_indexes = ['abstract','author','caption','equation','figure','footer','list','paragraph','reference','section','table','title','date'] + ["null"]
# # print(pd.DataFrame(confusion_matrix, columns=matrix_indexes, index=matrix_indexes))
