from __future__ import annotations
import logging
from collections import OrderedDict
import os, json, cv2, random, shutil
from tqdm import tqdm
from PIL import Image
import numpy as np

from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.structures import Boxes
import torch

# logger = setup_logger("Util Logger")


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def create_model(cfg, type="train"):
    if type == "train":
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
        return trainer
    if type == "test":
        tester = DefaultPredictor(cfg)
        return tester


def crop_object(image, box, ground_truth=False):
    """Crops an object in an image

  Inputs:
    image: PIL image
    box: one box from Detectron2 pred_boxes
    """
    if (not ground_truth):
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]
    else:
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[0] + box[2]
        y_bottom_right = box[1] + box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    try:
        crop_img = image.crop((int(x_top_left), int(y_top_left),
                               int(x_bottom_right), int(y_bottom_right)))
    except Exception as e:
        pass

    return crop_img


def do_evaluate(cfg, model, output_path):
    # logger=logging.getLogger(cfg.TRAINING_NAME)
    # logger.log("Evaluating")
    results = dict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name,
                                  output_dir=os.path.join(
                                      output_path, "inference", dataset_name))
        results_i = inference_on_dataset(model.model, data_loader, evaluator)
        results[dataset_name] = results_i
    return results


def remove_dataset(name):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)


'''
Returns the list of cropped image based on the objects. The method uses the trained object detection\
     model to get bouding box and crop the images.
'''


def crop_images_classwise(model: DefaultPredictor, src_path, dest_path,
                          proposal_budget: int):
    if not os.path.exists(dest_path + '/obj_images'):
        os.makedirs(dest_path + '/obj_images')
    obj_im_dir = dest_path + '/obj_images'
    # MAPPING = {
    #         "0":"aeroplane", "1":"bicycle", "2":"bird", "3":"boat", "4":"bottle", "5":"bus", "6":"car", "7":"cat",
    #         "8":"chair", "9":"cow", "10":"diningtable", "11":"dog", "12":"horse", "13":"motorbike", "14":"person",
    #         "15":"pottedplant", "16":"sheep", "17":"sofa", "18":"train", "19":"tvmonitor"}
    # MAPPING = {
    #     "0": "text",
    #     "1": "title",
    #     "2": "list",
    #     "3": "table",
    #     "4": "figure"
    # }
    MAPPING = {"0":"bg", "1":"Image","2":"Math", "3":"Table", "4":"Text"}
    no_of_objects = 0
    for d in tqdm(os.listdir(src_path)):
        image = cv2.imread(os.path.join(src_path, d))
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        images = model.model.preprocess_image(inputs)

        features = model.model.backbone(images.tensor)
        proposals, _ = model.model.proposal_generator(images, features)
        instances, _ = model.model.roi_heads(images, features,
                                                     proposals)
        # print(outputs["instances"].pred_classes.cpu().numpy().tolist())
        boxes = instances[0].pred_boxes
        classes = instances[0].pred_classes.cpu().numpy().tolist()
        max_score_order = torch.argsort(instances[0].scores).tolist()
        
        if (proposal_budget > len(max_score_order)):
            proposal_budget = len(max_score_order)
        
        for singleclass in classes:
            if not os.path.exists(
                    os.path.join(dest_path, 'obj_images',
                                 MAPPING[str(singleclass)])):
                os.makedirs(
                    os.path.join(dest_path, 'obj_images',
                                 MAPPING[str(singleclass)]))

        img = Image.open(os.path.join(src_path, d))
        for idx, box in enumerate(
                list(boxes[max_score_order[:proposal_budget]])):
            no_of_objects += 1
            box = box.detach().cpu().numpy()

            crop_img = crop_object(img, box)
            try:
                crop_img.save(
                    os.path.join(
                        obj_im_dir, MAPPING[str(classes[idx])],
                        os.path.split(os.path.join(src_path, d))[1].replace(
                            ".jpg", "") + "_" + str(idx) + ".jpg"))
            except Exception as e:
                print(e)

    print("Number of objects: " + str(no_of_objects))


'''
Returns the list of cropped images based on the objects. The method make use of ground truth to crop the image.
'''


def crop_images_classwise_ground_truth(train_json_path, src_path, dest_path,
                                       category: list):
    if not os.path.exists(dest_path + '/obj_images'):
        os.makedirs(dest_path + '/obj_images')
    obj_im_dir = dest_path + '/obj_images'
    # MAPPING = {
    #         "aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, "bus":6, "car":7, "cat":8,
    # "chair":9, "cow":10, "diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
    # "pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20, "null":21}
    # MAPPING = {"text": 1, "title": 2, "list": 3, "table": 4, "figure": 5}
    MAPPING = {"bg":0, "Image":1, "Math":2, "Table":3, "Text":4}
    no_of_objects = 0
    with open(train_json_path) as f:
        data = json.load(f)
    annotations = data['annotations']
    file_names = os.listdir(src_path)
    file_ids = {
        x['id']: x['file_name']
        for x in data['images'] if x['file_name'] in file_names
    }
    for idx, d in tqdm(file_ids.items()):
        img = cv2.imread(os.path.join(src_path, d))
        if not os.path.exists(
                os.path.join(dest_path, 'obj_images', category[0])):
            os.makedirs(os.path.join(dest_path, 'obj_images', category[0]))

        img = Image.open(os.path.join(src_path, d))
        boxes = [
            x['bbox'] for x in annotations if x['image_id'] == idx
            and x['category_id'] == MAPPING[category[0]]
        ]
        for idx, box in enumerate(list(boxes)):
            no_of_objects += 1
            box = np.asarray(box, dtype=np.float32)

            crop_img = crop_object(img, box, True)
            crop_img.save(
                os.path.join(
                    obj_im_dir, category[0],
                    os.path.split(os.path.join(src_path, d))[1].replace(
                        ".jpg", "") + "_" + str(idx) + ".jpg"))

    print("Number of objects: " + str(no_of_objects))


''''
    Returns the embedding of the query images
    Uses ground truth to obtain the bouding box, and ROI pooler to obtain the bouding box features.
'''


def get_query_embedding(predictor: DefaultPredictor, image_dir: str,
                        target_class: str, train_json_path, device):
    embeddings = []
    final_im_list = []
    with torch.no_grad():
        # MAPPING = {"text": 0, "title": 2, "list": 3, "table": 4, "figure": 5}
        MAPPING = {"bg":0, "Image":1, "Math":2, "Table":3, "Text":4}

        no_of_objects = 0
        with open(train_json_path) as f:
            data = json.load(f)
        annotations = data['annotations']
        file_names = os.listdir(image_dir)
        file_ids = {
            x['id']: x['file_name']
            for x in data['images'] if x['file_name'] in file_names
        }
        for idx, im in tqdm(file_ids.items()):
            image_path = os.path.join(image_dir, im)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            images = predictor.model.preprocess_image(inputs)

            features = predictor.model.backbone(images.tensor)

            boxes = [
                x['bbox'] for x in annotations if x['image_id'] == idx
                and x['category_id'] == MAPPING[target_class]
            ]
            box_features = [
                features[f] for f in predictor.model.roi_heads.in_features
            ]
            box_features_from_pooler = predictor.model.roi_heads.box_pooler(
                box_features, [Boxes(torch.Tensor(boxes).to(device))])
            box_features_from_head = predictor.model.roi_heads.box_head(
                box_features_from_pooler)

            box_features_weights = []
            for i in range(len(boxes)):
                box_features_weights.append(
                    box_features_from_head[i, :].cpu().numpy())

            for img_roi_feature in box_features_weights:
                embeddings.append(img_roi_feature)
                final_im_list.append(im)
        del features, box_features, box_features_from_pooler, box_features_from_head
    return final_im_list, np.stack(embeddings)


'''
Getting embedding for the lake set.
Uses resnet 101 backbone to obtain the feature map,
ROI to get respective bouding box and
ROI pooler to get respective bounding box feature map.
'''


def  get_lake_embedding(predictor: DefaultPredictor, image_dir: str,
                       proposal_budget: int):
    embeddings = []
    final_im_list = []
    with torch.no_grad():
        for im in tqdm(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, im)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            images = predictor.model.preprocess_image(inputs)

            features = predictor.model.backbone(images.tensor)
            proposals, _ = predictor.model.proposal_generator(images, features)
            instances, _ = predictor.model.roi_heads(images, features,
                                                     proposals)
            pred_classes = instances[0].pred_classes.cpu().numpy().tolist()
            max_score_order = torch.argsort(instances[0].scores).tolist()

            if (proposal_budget > len(max_score_order)):
                proposal_budget = len(max_score_order)

            box_features = [
                features[f] for f in predictor.model.roi_heads.in_features
            ]
            box_features_from_pooler = predictor.model.roi_heads.box_pooler(
                box_features, [x.pred_boxes for x in instances])
            box_features_from_head = predictor.model.roi_heads.box_head(
                box_features_from_pooler)

            for i in max_score_order[:proposal_budget]:
                embeddings.append(box_features_from_head[i, :].cpu().numpy())
                final_im_list.append(im)
        del features, proposals, instances, pred_classes, box_features, box_features_from_pooler, box_features_from_head
    return final_im_list, np.stack(embeddings)
