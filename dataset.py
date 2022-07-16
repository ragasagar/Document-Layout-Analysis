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
from detectron2.data import MetadataCatalog, DatasetCatalog
import shutil
from docbank_loader import DocBankLoader, DocBankConverter


from utils.utils import *
from utils.coco_util import *

def create_dataset():
    img_path = 'DocBank_500K_ori_img'
    txt_anno_path = 'DocBank_500K_txt'
    arrange_data("DocBank/DocBank_samples/DocBank_samples", img_path, txt_anno_path)



    txt_dir = 'txt'
    img_dir = 'img'
    # loader = DocBankLoader(txt_dir=txt_dir, img_dir=img_dir)

    # converter = DocBankConverter(loader)

    source_dirs = ("DocBank_500K_ori_img", "DocBank_500K_txt")
    # source_dirs = ("img", "txt")
    query_data_dirs = ("new_query_data_img", "new_query_data_txt_anno")
    train_data_dirs = ("train_data_img", "train_data_txt_anno")
    lake_data_dirs = ("lake_data_img", "lake_data_txt_anno")
    test_data_dirs = ("test_data_img", "test_data_txt_anno")
    val_data_dirs = ("val_data_img", "val_data_txt_anno")
    split_cfg = {
        "train_ratio":0.1,
        "val_ratio":0.1,
        "test_ratio":0.1

    }
    split_data(source_dirs, 
            train_dirs= train_data_dirs, 
            query_dirs=query_data_dirs,
            lake_dirs= lake_data_dirs,
            test_dirs=test_data_dirs, 
            val_dirs=val_data_dirs, 
            split_cfg=split_cfg
    )


    query_data_loader = DocBankLoader(txt_dir=query_data_dirs[1], img_dir=query_data_dirs[0])
    query_data_converter = DocBankConverter(query_data_loader)
    query_data_examples = query_data_converter.read_all()

    query_data_coco = COCOData("COCOQueryData.json",query_data_examples)
    query_data_coco.convert_to_coco()
    query_data_coco.save_coco_dataset()

    train_data_loader = DocBankLoader(txt_dir=train_data_dirs[1], img_dir=train_data_dirs[0])
    train_data_converter = DocBankConverter(train_data_loader)
    train_data_examples = train_data_converter.read_all()

    train_data_coco = COCOData("COCOTrainData.json",train_data_examples)
    train_data_coco.convert_to_coco()
    train_data_coco.save_coco_dataset()

    lake_data_loader = DocBankLoader(txt_dir=lake_data_dirs[1], img_dir=lake_data_dirs[0])
    lake_data_converter = DocBankConverter(lake_data_loader)
    lake_data_examples = lake_data_converter.read_all()

    lake_data_coco = COCOData("COCOLakeData.json",lake_data_examples)
    lake_data_coco.convert_to_coco()
    lake_data_coco.save_coco_dataset()


# create_dataset()