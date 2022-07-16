# Some basic setup:
# Setup detectron2 logger
from genericpath import exists
from tkinter import Image
from dataset import create_dataset
import detectron2
from detectron2.utils.logger import setup_logger
from target_feature_selection import ImageModel
from utils.coco_util import COCOData
from utils.submodlib import COM_wrapper, FL1CMI_wrapper, FL1MI_wrapper, Random_wrapper, subset
from utils.utils import find_missclassified_object
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, shutil
import torch

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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

setup_logger()
# from modelling.backbone import fpn
import cv2
import torch
from docbank_loader import DocBankLoader, DocBankConverter

def getEmbeddings(emb, lake_img_dir, query_img_dir, misclassified_object):
    
    '''query embeddings'''
    query_image_list, query_embeddings = emb.get_embeddings_image_list(query_img_dir, misclassified_object, is_query=False, sel_strategy='area')

    '''Lake embeddings'''
    lake_image_list, lake_embeddings = emb.get_embeddings_image_list(lake_img_dir, misclassified_object, is_query=False, sel_strategy='area')
    # print(len(lake_embeddings))
    # print(len(lake_image_list))
    return query_image_list, query_embeddings, lake_image_list, lake_embeddings;

def getEmbeddings_private(emb, private_img_dir, private_objects):
    private_image_list, private_embeddings = emb.get_embeddings_image_list(private_img_dir, private_objects, is_query=True)
    return private_image_list, private_embeddings;

def create_data_loader(query_data_dirs, lake_data_dirs, isTrain=False):    
    lake_data_loader = DocBankLoader(txt_dir=lake_data_dirs[1], img_dir=lake_data_dirs[0])
    lake_data_converter = DocBankConverter(lake_data_loader)
    lake_data_examples = lake_data_converter.read_all()

    lake_data_coco = COCOData("COCOLakeData.json",lake_data_examples)
    lake_data_coco.convert_to_coco()
    lake_data_coco.save_coco_dataset()

    if(isTrain):
        query_data_loader = DocBankLoader(txt_dir=query_data_dirs[1], img_dir=query_data_dirs[0])
        query_data_converter = DocBankConverter(query_data_loader)
        query_data_examples = query_data_converter.read_all()

        query_data_coco = COCOData("COCOQueryData.json",query_data_examples)
        query_data_coco.convert_to_coco()
        query_data_coco.save_coco_dataset()
    else:
        query_data_loader = DocBankLoader(txt_dir=query_data_dirs[1], img_dir=query_data_dirs[0])
        query_data_converter = DocBankConverter(query_data_loader)
        query_data_examples = query_data_converter.read_all()

        query_data_coco = COCOData("COCOTrainData.json",query_data_examples)
        query_data_coco.convert_to_coco()
        query_data_coco.save_coco_dataset()

def create_val_data_loader(val_data_dirs):
    lake_data_loader = DocBankLoader(txt_dir=val_data_dirs[1], img_dir=val_data_dirs[0])
    lake_data_converter = DocBankConverter(lake_data_loader)
    lake_data_examples = lake_data_converter.read_all()

    lake_data_coco = COCOData("COCOValData.json",lake_data_examples)
    lake_data_coco.convert_to_coco()
    lake_data_coco.save_coco_dataset()

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
    create_data_loader(query_data_dirs, lake_data_dir, True)

def remove_dataset(name):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

def create_model(cfg, type="train"):
    if type == "train":
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        return trainer;
    else:
        print(cfg)
        return ImageModel(cfg);

def main():

    source_dirs = ("DocBank_500K_ori_img", "DocBank_500K_txt")
    query_data_dirs = ("query_data_img", "query_data_txt_anno")
    train_data_dirs = ("train_data_img", "train_data_txt_anno")
    lake_data_dirs = ("lake_data_img", "lake_data_txt_anno")
    test_data_dirs = ("test_data_img", "test_data_txt_anno")
    val_data_dirs = ("val_data_img", "val_data_txt_anno")


    create_dataset();
    create_val_data_loader(val_data_dirs)

    ## initial lake image data path
    lake_image_list = os.listdir("lake_data_img")
    lake_image_list = [os.path.join("lake_data_img",x) for x in lake_image_list]

    ## query imae data path
    query_image_list = os.listdir("query_data_img")
    query_image_list = [os.path.join("query_data_img",x) for x in query_image_list]


    #initial weight and config path
    config_file_path = 'X101/X101.yaml'
    # weights_path = 'X101/model.pth';


    ## intial initializeation of the parameters
    register_coco_instances("docbank_seg_train",{}, "COCOTrainData.json", ".")
    # print(MetadataCatalog.get("train_data_coco"))
    register_coco_instances("docbank_seg_test",{}, "COCOValData.json", ".")
    # print(MetadataCatalog.get("val_data_coco"))
    # register_coco_instances("test_data_coco",{}, "COCOTestData.json", "test_data_img")
    
    budget = 200;
    selection_budget = 20;
    output_dir="new_model_weights"
    number_of_rounds = 10
    selection_strag = "fl1mi";
    private_set = False;

    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("docbank_seg_train",)
    cfg.DATASETS.TEST = ("docbank_seg_test",)
    cfg.OUTPUT_DIR = output_dir;

    model = create_model(cfg, "test");


    for i in range(0,number_of_rounds):
        if(i==0):
            print('Initial phase  \n Loading trained model')        
            cfg.merge_from_file(config_file_path)
            # cfg.MODEL.DEVICE = "cpu"
            # cfg.MODEL.WEIGHTS = weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

            if(not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))):
                model = create_model(cfg, "train")
                model.train()
            
            ## evaluating the model perfromance
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

            ## delete old predictor to free some cuda memory
            del model
            model = create_model(cfg, "test")

            
            evaluator = COCOEvaluator("docbank_seg_test", output_dir=output_dir)
            val_loader = build_detection_test_loader(cfg, "docbank_seg_test")
            result = inference_on_dataset(model.predictor.model, val_loader, evaluator)
        else:
            print("training start iteration:", i)
            if(model):
                del model
            cfg.merge_from_file(config_file_path)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
            model = create_model(cfg,"test")


            #finding the most misclassified objects from the models.
            query_path, category = find_missclassified_object(result);
            print(query_path,category);
            # finding the embedding of the boxes.
            if(i>5):
                _, query_embeddings, lake_image_list, lake_embeddings= getEmbeddings(model, "lake_data_img", query_path, category)
            else:
                _, query_embeddings, lake_image_list, lake_embeddings= getEmbeddings(model, "lake_data_img", 'query_data_img/equation', ['equation'])
            print(len(query_embeddings))

            '''Lake embeddings'''
            # print(len(lake_embeddings))
            print(len(lake_image_list))

            if(selection_strag !="random"):
                if(private_set):
                    _, private_embeddings = getEmbeddings(model, "query_data_img/figure", ["figure"]);
                subset_result = subset(lake_embeddings, query_embeddings, 2, lake_image_list, budget = selection_budget, metric = 'cosine', 
                                    stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, strategry=selection_strag)
            else:
                lake_image_list = os.listdir(lake_data_dirs[0])
                subset_result = Random_wrapper(lake_image_list, 10)
            
            budget-=selection_budget;
            if(budget>0):
                change_dir(subset_result, train_data_dirs, lake_data_dirs, budget)
            # Adding the data to the query image list
            lake_image_list = os.listdir("lake_data_img")
            lake_image_list = [os.path.join("lake_data_img",x) for x in lake_image_list]
            
            # Creating new embedding for the new dataset.
            train_image_list = os.listdir("train_data_img")
            train_image_list = [os.path.join("train_data_img",x) for x in train_image_list]

            remove_dataset("train_data_coco")
            register_coco_instances("train_data_coco",{}, "COCOTrainData.json", "train_data_img")
            if(model):
                del model
            model = create_model(cfg)
            model.train()

            ## evaluating the model perfromance
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

            ## delete old predictor to free some cuda memory
            del model
            model = create_model(cfg, "test")
            evaluator = COCOEvaluator("docbank_seg_test", output_dir=output_dir)
            val_loader = build_detection_test_loader(cfg, "docbank_seg_test")
            result = inference_on_dataset(model.predictor.model, val_loader, evaluator)
            # print("showing result")
            
    print("completed result");



if __name__ == "__main__":
    main();

    
