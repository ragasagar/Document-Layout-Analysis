
from target_feature_selection import Targeted_Feature_Selection
from utils.submodlib import submod_results
from utils.utils import *
from utils.coco_util import *
from dataset import create_dataset

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# import some common libraries
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2, os



create_dataset();

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "X101/model.pth" 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7


print(cfg)
# Opening JSON file
f = open('COCOLakeData.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
category_list = [cat['name'] for cat in data['categories']]
lake_images = [image["file_name"] for image in data["images"]]
lake_images_area = {image["file_name"]: image["height"]*image["width"] for image in data["images"]}
lake_annotation_dict = {im:[] for im in lake_images}


for ann in data["annotations"]:
    image_index = ann["image_id"]
    lake_annotation_dict[lake_images[image_index]].append([ann["area"],ann["category_id"]+1 ])


f = open('COCOQueryData.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
# category_list = [cat['name'] for cat in data['categories']]
query_images = [image["file_name"] for image in data["images"]]
query_images_area = {image["file_name"]: image["height"]*image["width"] for image in data["images"]}
query_annotation_dict = {im:[] for im in query_images}


for ann in data["annotations"]:
    image_index = ann["image_id"]
    query_annotation_dict[query_images[image_index]].append([ann["area"],ann["category_id"]+1 ])



np.random.seed(42)
index = np.random.randint(len(os.listdir("query_data_img")))
image_name = os.listdir("query_data_img")[index]
image = cv2.imread(os.path.join("query_data_img", image_name))
# cv2.imshow('sample image',image) 
# cv2.waitKey(0)

 
# Opening JSON file
with open('COCOQueryData.json') as json_file:
    ground_data = json.load(json_file)
 
for im in ground_data["images"]:
    if im["file_name"]==os.path.join("query_data_img", image_name):
        image_id = im["id"]
        break

annotations = []

for i,annotation in enumerate(ground_data["annotations"]):
    if annotation["image_id"]==image_id:
        annotations.append(annotation["bbox"])

targeted_selector = Targeted_Feature_Selection(cfg, "X101/model.pth", category_list=category_list)

lake_image_list = os.listdir("lake_data_img")
lake_image_list = [os.path.join("lake_data_img",x) for x in lake_image_list]

query_image_list = os.listdir("query_data_img")
query_image_list = [os.path.join("query_data_img",x) for x in query_image_list]



## Strategy 1 - Average

''' Divide region propsals based on labels and simply take their average.''' 
lake_data_avg, lake_dataset = targeted_selector.targeted_classes_image_embeddings(lake_image_list, sel_strategy = "avg")
query_data_avg, query_dataset = targeted_selector.targeted_classes_image_embeddings(query_image_list, sel_strategy = "avg")

percent_dict = {"Random":[],"FL1MI":[], "FL2MI":[], "COM":[], "GCMI":[], "LogDetMI":[]}

etas = [0, 0.3, 0.6, 1, 1.4, 1.8, 2.2, 2.6, 3, 10, 50, 100]

for eta in etas:   
    Random_percent, FL1MI_percent, FL2MI_percent, COM_percent, GCMI_percent, LogDetMI_percent = submod_results(lake_data_avg, query_data_avg, eta, targeted_classes=['Equation','Table'], image_list=lake_image_list, annotations=lake_annotation_dict, image_areas=lake_images_area, category_list=category_list, budget = 10, metric='cosine')
    percent_dict["Random"].append(Random_percent)
    percent_dict["FL1MI"].append(FL1MI_percent)
    percent_dict["FL2MI"].append(FL2MI_percent)
    percent_dict["COM"].append(COM_percent)
    percent_dict["GCMI"].append(GCMI_percent)
    percent_dict["LogDetMI"].append(LogDetMI_percent)

### Plot max percentage vs $\eta$ 
Random_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["Random"]]
FL1MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL1MI"]]
FL2MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL2MI"]]
COM_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["COM"]]
GCMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["GCMI"]]
LogDetMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["LogDetMI"]]

plt.plot(etas, Random_max, label = "Random")
plt.plot(etas, FL1MI_max, label = "FL1MI")
plt.plot(etas, FL2MI_max, label = "FL2MI")
plt.plot(etas, COM_max, label = "COM")
plt.plot(etas, GCMI_max, label = "GCMI")
plt.plot(etas, LogDetMI_max, label = "LogDetMI")
plt.xscale("log")
plt.legend()
plt.show()

## Strategy 2 - Weighted Average

# Divide region propsals based on labels and take their average based on the probability score assigned by the network.

lake_data_avg, lake_dataset = targeted_selector.targeted_classes_image_embeddings(lake_image_list, sel_strategy = "weight_avg")
query_data_avg, query_dataset = targeted_selector.targeted_classes_image_embeddings(query_image_list, sel_strategy = "weight_avg")

print(lake_data_avg.shape, len(lake_dataset), query_data_avg.shape, len(query_dataset))

percent_dict = {"Random":[],"FL1MI":[], "FL2MI":[], "COM":[], "GCMI":[], "LogDetMI":[]}

for eta in etas:   
    Random_percent, FL1MI_percent, FL2MI_percent, COM_percent, GCMI_percent, LogDetMI_percent = submod_results(lake_data_avg, query_data_avg, eta, targeted_classes=['Equation','Table'], image_list=lake_image_list, annotations=lake_annotation_dict, image_areas=lake_images_area, category_list=category_list, budget = 10, metric='cosine')
    percent_dict["Random"].append(Random_percent)
    percent_dict["FL1MI"].append(FL1MI_percent)
    percent_dict["FL2MI"].append(FL2MI_percent)
    percent_dict["COM"].append(COM_percent)
    percent_dict["GCMI"].append(GCMI_percent)
    percent_dict["LogDetMI"].append(LogDetMI_percent)

Random_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["Random"]]
FL1MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL1MI"]]
FL2MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL2MI"]]
COM_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["COM"]]
GCMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["GCMI"]]
LogDetMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["LogDetMI"]]


plt.plot(etas, Random_max, label = "Random")
plt.plot(etas, FL1MI_max, label = "FL1MI")
plt.plot(etas, FL2MI_max, label = "FL2MI")
plt.plot(etas, COM_max, label = "COM")
plt.plot(etas, GCMI_max, label = "GCMI")
plt.plot(etas, LogDetMI_max, label = "LogDetMI")
plt.xscale("log")
plt.legend()
plt.show()

## Strategy 3 - Maximum area bounding box

# Among all bounding boxes predicted, find the one with the maximum area among the targeted classes.

lake_data_area, lake_dataset = targeted_selector.targeted_classes_image_embeddings(lake_image_list, sel_strategy = "area")
query_data_area, query_dataset = targeted_selector.targeted_classes_image_embeddings(query_image_list, sel_strategy = "area")


percent_dict = {"Random":[],"FL1MI":[], "FL2MI":[], "COM":[], "GCMI":[], "LogDetMI":[]}

for eta in etas:   
    random_percent, FL1MI_percent, FL2MI_percent, COM_percent, GCMI_percent, LogDetMI_percent = submod_results(lake_data_area, query_data_area, eta, targeted_classes=['Equation','Table'], image_list=lake_image_list, annotations=lake_annotation_dict, image_areas=lake_images_area, category_list=category_list, budget = 10, metric='cosine')
    percent_dict["FL1MI"].append(FL1MI_percent)
    percent_dict["Random"].append(Random_percent)
    percent_dict["FL2MI"].append(FL2MI_percent)
    percent_dict["COM"].append(COM_percent)
    percent_dict["GCMI"].append(GCMI_percent)
    percent_dict["LogDetMI"].append(LogDetMI_percent)

Random_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["Random"]]
FL1MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL1MI"]]
FL2MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL2MI"]]
COM_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["COM"]]
GCMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["GCMI"]]
LogDetMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["LogDetMI"]]


plt.plot(etas, Random_max, label = "Random")
plt.plot(etas, FL1MI_max, label = "FL1MI")
plt.plot(etas, FL2MI_max, label = "FL2MI")
plt.plot(etas, COM_max, label = "COM")
plt.plot(etas, GCMI_max, label = "GCMI")
plt.plot(etas, LogDetMI_max, label = "LogDetMI")
plt.xscale("log")
plt.legend()
plt.show()


## Strategy 4 - Max score bounding box

# Among all bounding boxes predicted, find the one with the maximum score among the targeted classes.
lake_data_score, lake_dataset = targeted_selector.targeted_classes_image_embeddings(lake_image_list, sel_strategy = "score")
query_data_score, query_dataset = targeted_selector.targeted_classes_image_embeddings(query_image_list, sel_strategy = "score")

percent_dict = {"Random":[],"FL1MI":[], "FL2MI":[], "COM":[], "GCMI":[], "LogDetMI":[]}

for eta in etas:   
    Random_percent, FL1MI_percent, FL2MI_percent, COM_percent, GCMI_percent, LogDetMI_percent = submod_results(lake_data_score, query_data_score, eta, targeted_classes=['Equation','Table'], image_list=lake_image_list, annotations=lake_annotation_dict, image_areas=lake_images_area, category_list=category_list, budget = 10, metric='cosine')
    percent_dict["FL1MI"].append(FL1MI_percent)
    percent_dict["Random"].append(Random_percent)
    percent_dict["FL2MI"].append(FL2MI_percent)
    percent_dict["COM"].append(COM_percent)
    percent_dict["GCMI"].append(GCMI_percent)
    percent_dict["LogDetMI"].append(LogDetMI_percent)

Random_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["Random"]]
FL1MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL1MI"]]
FL2MI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["FL2MI"]]
COM_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["COM"]]
GCMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["GCMI"]]
LogDetMI_max = [max(l)*100 if len(l)>0 else 0 for l in percent_dict["LogDetMI"]]


# plt.plot(etas, Random_max, label = "Random")
plt.plot(etas, FL1MI_max, label = "FL1MI")
plt.plot(etas, FL2MI_max, label = "FL2MI")
plt.plot(etas, COM_max, label = "COM")
plt.plot(etas, GCMI_max, label = "GCMI")
plt.plot(etas, LogDetMI_max, label = "LogDetMI")
plt.xscale("log")
plt.legend()
plt.show()