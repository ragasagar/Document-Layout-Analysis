# Some basic setup:
# Setup detectron2 logger
from genericpath import exists
from re import T
from tkinter import Image
from dataset import create_dataset
import detectron2
from detectron2.utils.logger import setup_logger
from feature_selection import ImageModel
from utils.coco_util import COCOData
from utils.submodlib import COM_wrapper, FL1CMI_wrapper, FL1MI_wrapper, Random_wrapper, subset
from utils.utils import aug_train_subset, find_missclassified_object, get_test_score
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, shutil
import torch
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

setup_logger()
# from modelling.backbone import fpn
import cv2
import torch
from docbank_loader import DocBankLoader, DocBankConverter

def getEmbeddings(emb, lake_img_dir, query_img_dir, misclassified_object):
    
    '''query embeddings'''
    query_image_list, query_embeddings = emb.get_embeddings_image_list(query_img_dir, misclassified_object, is_query=False, sel_strategy='weight_avg')

    '''Lake embeddings'''
    lake_image_list, lake_embeddings = emb.get_embeddings_image_list(lake_img_dir, misclassified_object, is_query=False, sel_strategy='weight_avg')
    # print(len(lake_embeddings))
    # print(len(lake_image_list))
    return query_image_list, query_embeddings, lake_image_list, lake_embeddings;

def getEmbeddings_private(emb, private_img_dir, private_objects):
    private_image_list, private_embeddings = emb.get_embeddings_image_list(private_img_dir, private_objects, is_query=True)
    return private_image_list, private_embeddings;

def create_data_loader(data_dirs, filename):
    data_loader = DocBankLoader(txt_dir=data_dirs[1], img_dir=data_dirs[0])
    data_converter = DocBankConverter(data_loader)
    data_examples = data_converter.read_all()

    data_coco = COCOData(filename, data_examples)
    data_coco.convert_to_coco()
    data_coco.save_coco_dataset()

def change_dir(image_results, query_data_dirs, lake_data_dir, budget):
    names = [names.split("/")[-1].replace("_ori.jpg", "") for names in image_results]
    for index in range(len(names)):
        # Source path
        source_img = os.path.join(lake_data_dir[0], "{}_ori.jpg".format(names[index]))
        # Destination path
        source_txt = os.path.join(lake_data_dir[1], "{}.txt".format(names[index]))

        destination_img = os.path.join(query_data_dirs[0], "{}_ori.jpg".format(names[index]))
        # Destination path
        destination_txt = os.path.join(query_data_dirs[1], "{}.txt".format(names[index]))
        
        if not os.path.exists(query_data_dirs[0]) or not os.path.exists(query_data_dirs[1]):
            os.mkdir(query_data_dirs[0])
            os.mkdir(query_data_dirs[1])
        
        try:
            shutil.copy(source_img, destination_img)
            shutil.copy(source_txt, destination_txt)
            # print("File copied successfully.")
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        
        # For other errors
        except:
            print("Error occurred while copying file.")


        # removing the data from the lake data
        try:
            os.remove(os.path.join(lake_data_dir[0], "{}_ori.jpg".format(names[index])))
            os.remove(os.path.join(lake_data_dir[1], "{}.txt".format(names[index])))
        except:
            pass


    #loading new data loader and saving coco
    create_data_loader(lake_data_dir, 'COCOTrainData.json')

def remove_dataset(name):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

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
        return trainer;
    else:
        print(cfg)
        return ImageModel(cfg);


def main():

    train_data_dirs = ("train_data_img", "PASCAL_VOC/PASCAL_VOC/train_targeted.json")
    lake_data_dirs = ("lake_data_img", "PASCAL_VOC/PASCAL_VOC/lake_targeted.json")
    test_data_dirs = ("test_data_img", "PASCAL_VOC/PASCAL_VOC/val_targeted.json")
    val_data_dirs = ("val_data_img", "PASCAL_VOC/PASCAL_VOC/test_targeted.json")


    create_dataset()
    
    ## intial initializeation of the parameters
    register_coco_instances("docbank_seg_train",{}, "PASCAL_VOC/PASCAL_VOC/train_targeted.json", "train_data_img")
    register_coco_instances("docbank_seg_val",{}, "PASCAL_VOC/PASCAL_VOC/val_targeted.json", "val_data_img")
    register_coco_instances("docbank_seg_test",{}, "PASCAL_VOC/PASCAL_VOC/test_targeted.json", "test_data_img")
    
    budget = 200;
    selection_budget = 20;
    output_dir="dynamic_weight_change"
    trained_model_weights_path = "coco-weights/model_final_coco.pkl";
    number_of_rounds = 100
    selection_strag = "fl1mi";
    private_set = False;
    
    #initial weight and config path
    config_file_path = 'configs/coco/faster_rcnn_R_101_FPN_3x.yaml'
    prediction_score_threshold = 0.7
    # Detectron config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("docbank_seg_train",)
    cfg.DATASETS.TEST = ('docbank_seg_val',)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold   # set a custom testing threshold
    cfg.OUTPUT_DIR = output_dir;
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01

    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 2000 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, 1900)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21 #your number of classes + 1
    cfg.TEST.EVAL_PERIOD = 500

    final_test_score = [];
    final_val_score = [];
    test_score = 0;
    iteration = 0;
    try:
        while(test_score < 90 and iteration < number_of_rounds):    
            if(iteration==0):
                print('Initial phase  \n Loading trained model')        
                if(not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))):
                    cfg.MODEL.WEIGHTS = trained_model_weights_path
                    model = create_model(cfg, "train")
                    model.train()
                
                ## evaluating the model perfromance
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

                model = create_model(cfg, "test")

                evaluator = COCOEvaluator("docbank_seg_val", output_dir=output_dir)
                val_loader = build_detection_test_loader(cfg, "docbank_seg_val")
                val_result = inference_on_dataset(model.predictor.model, val_loader, evaluator)

                evaluator = COCOEvaluator("docbank_seg_test", output_dir=output_dir)
                val_loader = build_detection_test_loader(cfg, "docbank_seg_test")
                result = inference_on_dataset(model.predictor.model, val_loader, evaluator)
            else:
                print("training start iteration:", iteration)
                
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
                model = create_model(cfg,"test")


                #finding the most misclassified objects from the models.
                query_path, category = find_missclassified_object(result);
                print(query_path,category);
                # finding the embedding of the boxes.
                # if(iteration>5):
                _, query_embeddings, lake_image_list, lake_embeddings= getEmbeddings(model, "lake_data_img", query_path, category)
                # else:
                #     _, query_embeddings, lake_image_list, lake_embeddings= getEmbeddings(model, "lake_data_img", 'query_data_img/aeroplane', ['aeroplane'])
                print(len(query_embeddings))

                '''Lake embeddings'''
                # print(len(lake_embeddings))
                print(len(lake_image_list))

                if(selection_strag !="random"):
                    # if(private_set):
                    #     _, private_embeddings = getEmbeddings(model, "query_data_img/figure", ["figure"])
                    subset_result = subset(lake_embeddings, query_embeddings, 2, lake_image_list, budget = selection_budget, metric = 'cosine', 
                                        stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, strategry=selection_strag)
                else:
                    lake_image_list = os.listdir(lake_data_dirs[0])
                    subset_result = Random_wrapper(lake_image_list, 10)
                
                budget-=selection_budget;
                if(budget>0):
                    aug_train_subset(subset_result, train_data_dirs[1], lake_data_dirs[1], budget, lake_data_dirs, train_data_dirs)
                # Adding the data to the query image list
                lake_image_list = os.listdir("lake_data_img")
                lake_image_list = [os.path.join("lake_data_img",x) for x in lake_image_list]
                
                # Creating new embedding for the new dataset.
                train_image_list = os.listdir("train_data_img")
                train_image_list = [os.path.join("train_data_img",x) for x in train_image_list]

                remove_dataset("docbank_seg_train")
                register_coco_instances("docbank_seg_train",{}, "PASCAL_VOC/PASCAL_VOC/train_targeted.json", "train_data_img")
                
                model = create_model(cfg)
                model.train()

                ## evaluating the model perfromance
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
                
                model = create_model(cfg, "test")
                #evaluating val model.
                evaluator = COCOEvaluator("docbank_seg_val", output_dir=output_dir)
                val_loader = build_detection_test_loader(cfg, "docbank_seg_val")
                val_result = inference_on_dataset(model.predictor.model, val_loader, evaluator)

                #evaluationg test models
                evaluator = COCOEvaluator("docbank_seg_test", output_dir=output_dir)
                val_loader = build_detection_test_loader(cfg, "docbank_seg_test")
                result = inference_on_dataset(model.predictor.model, val_loader, evaluator)


                #putting each round val score and test cores.
                final_test_score.append(result)
                final_val_score.append(val_result)
                test_score = get_test_score(result)
            
            #increasing iteration.
            iteration+=1;
    except Exception as e:
        print("Exeception occur while training", e)
    finally:
        final_test_score.append(result)
        final_val_score.append (val_result)
        final_data=[];
        temp = []
        for i in final_val_score:
            print(i);
            for k , val  in  i.items():
                temp = list(val.keys())
                final_data.append(list(val.values()))
        csv = pd.DataFrame(final_data, columns=temp)
        csv.to_csv("val_scores.csv")
        final_data=[];
        for i in final_test_score:
            print(i)
            for k , val  in  i.items():
                temp = list(val.keys())
                final_data.append(list(val.values()))
        csv = pd.DataFrame(final_data, columns=temp)
        csv.to_csv("test_score.csv")
            
    print("completed result");



if __name__ == "__main__":
    main();